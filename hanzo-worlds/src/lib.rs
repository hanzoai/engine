//! Native active-inference boundary for `/v1/worlds/{world}/inference` hosts.
//!
//! The host authenticates identity, resolves the tenant-scoped world and supplies its
//! immutable versioned generative model, preferences, habits and hard admissibility
//! policy. Callers can submit explicit beliefs, optional observations and candidate
//! policies, but can never overwrite preferences, transition/observation tables,
//! tenant identity or admissibility constraints.
//!
//! Inference, advisory recommendation and execution authorization are strictly
//! separated: every [`Response`] reports mathematical inference outputs under
//! `inference`, an explicitly advisory [`Recommendation`] (`advisory_only: true`),
//! and `execution_authorized: false`. There is no actuator interface.
//!
//! The canonical math lives in `hanzoai/ml`'s `hanzo-active-inference` crate,
//! never copied here.

use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::sync::{Arc, RwLock};

use hanzo_active_inference::{Model, Policies, Posterior, Result};
use serde::{Deserialize, Serialize};

/// Construct a finite-state model without coupling consumers to a tensor backend.
pub use hanzo_active_inference::Model as GenerativeModel;
/// Re-export the one canonical active-inference types for in-process engine hosts.
pub use hanzo_active_inference::{Counts, Error, Terms};

/// Maximum candidate policies admitted in one inference request.
pub const MAX_POLICIES: usize = 256;
/// Maximum open-loop planning horizon (actions per policy) in one inference request.
pub const MAX_HORIZON: usize = 16;
/// Maximum length of world and model revision identifiers.
pub const MAX_ID_LEN: usize = 128;

/// Validate an identifier used for world IDs or model revisions.
/// Rejects empty strings, overly long names, path traversal (`..`), and non-ASCII-safe characters.
pub fn is_safe_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= MAX_ID_LEN
        && !id.contains("..")
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'.' | b':' | b'-'))
}

/// A caller request to filter beliefs and evaluate candidate policies.
///
/// `#[serde(deny_unknown_fields)]` prevents callers from injecting host-owned fields
/// such as `admissible`, `habits`, `preferences`, `precision`, `tenant_id`, or
/// `execution_authorized`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// Current prior belief distribution in the world's registered latent state space.
    pub belief: Vec<f64>,
    /// Optional current observation: `(likelihood_action, outcome)`.
    #[serde(default)]
    pub observation: Option<(usize, usize)>,
    /// Candidate open-loop action sequences; the host evaluates admissibility and habits.
    pub policies: Vec<Vec<usize>>,
}

/// Pure mathematical inference output: belief filtering and exact EFE policy posterior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceOutput {
    /// Filtered current state and exact variational free energy, if an observation was supplied.
    pub posterior: Option<Posterior>,
    /// Exact finite-state expected free-energy terms and policy probabilities.
    pub policies: Policies,
}

/// Strictly advisory policy ranking derived from inference.
///
/// Separated from both mathematical [`InferenceOutput`] and execution authorization;
/// a recommendation never authorizes actuation or external side effects.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Recommendation {
    /// Index of the selected policy in the caller's candidate list.
    pub policy_index: usize,
    /// Action sequence of the recommended policy.
    pub actions: Vec<usize>,
    /// Posterior probability `Q(pi*)` of the recommended policy.
    pub probability: f64,
    /// Expected free energy `G(pi*)` in nats.
    pub expected_free_energy: f64,
    /// Invariantly `true`: recommendations are advisory mathematical rankings only.
    pub advisory_only: bool,
}

/// World inference response with strict separation between inference, recommendation,
/// and execution authorization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// World identifier when resolved through a host [`WorldRegistry`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub world_id: Option<String>,
    /// Immutable revision of the host-owned generative model that produced this result.
    pub model_version: String,
    /// Mathematical inference output (Bayesian belief posterior + exact EFE policy posterior).
    pub inference: InferenceOutput,
    /// Strictly advisory policy recommendation (`advisory_only: true`).
    pub recommendation: Recommendation,
    /// Invariantly `false`: inference and recommendation never authorize execution.
    pub execution_authorized: bool,
}

/// Immutable versioned generative model and host-owned policy precision.
/// An instance belongs to one authenticated tenant/world revision.
#[derive(Debug, Clone)]
pub struct WorldInference {
    model: Model,
    precision: f64,
    model_version: String,
}

impl WorldInference {
    /// Load a validated generative model with default revision `"v1"`.
    /// Precision is validated at inference.
    pub fn new(model: Model, precision: f64) -> Self {
        Self {
            model,
            precision,
            model_version: "v1".to_owned(),
        }
    }

    /// Load a validated generative model bound to an explicit immutable model version.
    pub fn with_version(
        model: Model,
        precision: f64,
        model_version: impl Into<String>,
    ) -> Result<Self> {
        let version = model_version.into();
        if !is_safe_id(&version) {
            return Err(Error(
                "model_version must be 1..128 safe ASCII characters without '..'".into(),
            ));
        }
        if !precision.is_finite() || precision < 0.0 {
            return Err(Error("precision must be finite and non-negative".into()));
        }
        Ok(Self {
            model,
            precision,
            model_version: version,
        })
    }

    /// Immutable version identifier of this generative model.
    pub fn model_version(&self) -> &str {
        &self.model_version
    }

    /// Host-owned inverse temperature (policy precision).
    pub fn precision(&self) -> f64 {
        self.precision
    }

    /// Read-only access to the validated immutable generative model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Filter optional observation, evaluate exact EFE under host-owned habits and
    /// hard admissibility mask, and return strictly separated inference, advisory
    /// recommendation, and `execution_authorized: false`.
    pub fn infer(
        &self,
        request: &Request,
        habits: &[f64],
        admissible: &[bool],
    ) -> Result<Response> {
        if request.policies.is_empty() || request.policies.len() > MAX_POLICIES {
            return Err(Error("candidate policy count must be in 1..=256".into()));
        }
        if request
            .policies
            .iter()
            .any(|p| p.is_empty() || p.len() > MAX_HORIZON)
        {
            return Err(Error("each policy horizon must be in 1..=16".into()));
        }
        let posterior = request
            .observation
            .map(|(a, o)| self.model.observe(&request.belief, a, o))
            .transpose()?;
        let belief = posterior
            .as_ref()
            .map_or(request.belief.as_slice(), |p| p.belief.as_slice());
        let policies = self.model.policies(
            belief,
            &request.policies,
            habits,
            admissible,
            self.precision,
        )?;
        let selected = policies.selected;
        if selected >= request.policies.len() || !admissible.get(selected).copied().unwrap_or(false)
        {
            return Err(Error(
                "selected policy failed host admissibility invariant".into(),
            ));
        }
        let recommendation = Recommendation {
            policy_index: selected,
            actions: request.policies[selected].clone(),
            probability: policies.probabilities[selected],
            expected_free_energy: policies.terms[selected].expected_free_energy,
            advisory_only: true,
        };
        Ok(Response {
            world_id: None,
            model_version: self.model_version.clone(),
            inference: InferenceOutput {
                posterior,
                policies,
            },
            recommendation,
            execution_authorized: false,
        })
    }
}

/// Host-owned hard admissibility constraints evaluated before policy selection.
/// Callers cannot supply, relax, or override these constraints.
#[derive(Debug, Clone)]
pub struct AdmissibilityPolicy {
    allowed_actions: BTreeSet<usize>,
    forbidden_prefixes: Vec<Vec<usize>>,
    max_horizon: usize,
    max_policies: usize,
}

impl AdmissibilityPolicy {
    /// Permit only candidate policies whose actions all belong to `actions`.
    pub fn allow_actions(actions: impl IntoIterator<Item = usize>) -> Self {
        Self {
            allowed_actions: actions.into_iter().collect(),
            forbidden_prefixes: Vec::new(),
            max_horizon: MAX_HORIZON,
            max_policies: MAX_POLICIES,
        }
    }

    /// Forbid any policy starting with `prefix` (hard safety veto).
    pub fn with_forbidden_prefix(mut self, prefix: Vec<usize>) -> Self {
        if !prefix.is_empty() {
            self.forbidden_prefixes.push(prefix);
        }
        self
    }

    /// Cap the permitted planning horizon (`1..=MAX_HORIZON`).
    pub fn with_max_horizon(mut self, max_horizon: usize) -> Self {
        self.max_horizon = max_horizon.clamp(1, MAX_HORIZON);
        self
    }

    /// Cap the permitted number of candidate policies (`1..=MAX_POLICIES`).
    pub fn with_max_policies(mut self, max_policies: usize) -> Self {
        self.max_policies = max_policies.clamp(1, MAX_POLICIES);
        self
    }

    /// Evaluate structural request bounds and compute the hard host admissibility mask.
    pub fn evaluate(
        &self,
        policies: &[Vec<usize>],
    ) -> std::result::Result<Vec<bool>, RegistryError> {
        if policies.is_empty() || policies.len() > self.max_policies {
            return Err(RegistryError::MalformedInput(format!(
                "candidate policy count must be in 1..={}",
                self.max_policies
            )));
        }
        for policy in policies {
            if policy.is_empty() || policy.len() > self.max_horizon {
                return Err(RegistryError::MalformedInput(format!(
                    "each policy horizon must be in 1..={}",
                    self.max_horizon
                )));
            }
        }
        let mask: Vec<bool> = policies
            .iter()
            .map(|policy| {
                policy.iter().all(|a| self.allowed_actions.contains(a))
                    && !self
                        .forbidden_prefixes
                        .iter()
                        .any(|prefix| policy.starts_with(prefix))
            })
            .collect();
        if !mask.iter().any(|&ok| ok) {
            return Err(RegistryError::Inadmissible(
                "no candidate policy satisfies host admissibility constraints".into(),
            ));
        }
        Ok(mask)
    }
}

/// Domain error returned by [`WorldRegistry`], separating auth, tenancy, validation,
/// and policy admissibility failures for HTTP/RPC hosts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Missing or invalid authenticated tenant identity (maps to HTTP 403).
    Unauthorized(String),
    /// World does not exist in the caller's tenant scope (maps to HTTP 404).
    /// Cross-tenant world IDs always return `NotFound` to prevent existence probing.
    NotFound,
    /// Malformed request shape, bad world ID, invalid distribution, or impossible observation (HTTP 400).
    MalformedInput(String),
    /// All candidate policies violate host admissibility constraints (HTTP 422).
    Inadmissible(String),
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unauthorized(msg) => write!(f, "unauthorized: {msg}"),
            Self::NotFound => write!(f, "world not found"),
            Self::MalformedInput(msg) => write!(f, "malformed input: {msg}"),
            Self::Inadmissible(msg) => write!(f, "inadmissible policies: {msg}"),
        }
    }
}

impl std::error::Error for RegistryError {}

#[derive(Debug, Clone)]
struct RegisteredWorld {
    engine: WorldInference,
    admissibility: AdmissibilityPolicy,
}

/// Thread-safe, multi-tenant host registry binding `(tenant_id, world_id)` to an
/// immutable versioned [`WorldInference`] model and host-owned [`AdmissibilityPolicy`].
///
/// Cross-tenant lookups always return [`RegistryError::NotFound`].
#[derive(Debug, Clone, Default)]
pub struct WorldRegistry {
    worlds: Arc<RwLock<HashMap<(String, String), RegisteredWorld>>>,
}

impl WorldRegistry {
    /// Create an empty fail-closed world registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Host-only registration of a versioned world model and hard admissibility policy
    /// under an authoritative `tenant_id` and `world_id`.
    pub fn register(
        &self,
        tenant_id: impl Into<String>,
        world_id: impl Into<String>,
        engine: WorldInference,
        admissibility: AdmissibilityPolicy,
    ) -> std::result::Result<(), RegistryError> {
        let tenant = tenant_id.into();
        let tenant = tenant.trim();
        if tenant.is_empty() || tenant.len() > MAX_ID_LEN {
            return Err(RegistryError::Unauthorized(
                "tenant_id must be non-empty and <= 128 bytes".into(),
            ));
        }
        let world = world_id.into();
        if !is_safe_id(&world) {
            return Err(RegistryError::MalformedInput("invalid world_id".into()));
        }
        if !engine.precision().is_finite() || engine.precision() < 0.0 {
            return Err(RegistryError::MalformedInput(
                "engine precision must be finite and non-negative".into(),
            ));
        }
        let mut guard = self
            .worlds
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.insert(
            (tenant.to_owned(), world),
            RegisteredWorld {
                engine,
                admissibility,
            },
        );
        Ok(())
    }

    /// Remove a registered world from `tenant_id`'s namespace.
    pub fn remove(&self, tenant_id: &str, world_id: &str) -> bool {
        let mut guard = self
            .worlds
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard
            .remove(&(tenant_id.trim().to_owned(), world_id.to_owned()))
            .is_some()
    }

    /// Total registered worlds across all tenants.
    pub fn len(&self) -> usize {
        self.worlds
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }

    /// Returns `true` when no worlds are registered.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Run tenant-isolated inference against `(tenant_id, world_id)`.
    ///
    /// - Rejects empty/missing tenant identity with [`RegistryError::Unauthorized`].
    /// - Rejects invalid/path-traversal `world_id` with [`RegistryError::MalformedInput`].
    /// - Returns [`RegistryError::NotFound`] if the world is absent or owned by another tenant.
    /// - Enforces host-owned admissibility constraints and uniform host policy priors.
    /// - Guarantees `response.recommendation.advisory_only == true` and
    ///   `response.execution_authorized == false`.
    pub fn infer(
        &self,
        tenant_id: &str,
        world_id: &str,
        request: &Request,
    ) -> std::result::Result<Response, RegistryError> {
        let tenant = tenant_id.trim();
        if tenant.is_empty() {
            return Err(RegistryError::Unauthorized(
                "authenticated tenant identity required".into(),
            ));
        }
        if !is_safe_id(world_id) {
            return Err(RegistryError::MalformedInput("invalid world id".into()));
        }
        let entry = {
            let guard = self
                .worlds
                .read()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard
                .get(&(tenant.to_owned(), world_id.to_owned()))
                .cloned()
                .ok_or(RegistryError::NotFound)?
        };
        let mask = entry.admissibility.evaluate(&request.policies)?;
        let n = request.policies.len();
        let habits = vec![1.0 / (n as f64); n];
        let mut response = entry
            .engine
            .infer(request, &habits, &mask)
            .map_err(|e| RegistryError::MalformedInput(e.0))?;
        response.world_id = Some(world_id.to_owned());
        response.recommendation.advisory_only = true;
        response.execution_authorized = false;
        Ok(response)
    }
}
