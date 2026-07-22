//! The policy seam: one trait, [`RoutePolicy`], that turns a request into a
//! [`Route`] (which model, at what level, in what modality, with what
//! confidence). This is the boundary that decomplects mechanism from brain:
//! `hanzo-router` owns the mechanism (registry, SLO gate, placement, dispatch);
//! a learned policy (the `enso` crate) owns the brain and implements this trait.
//!
//! The rule-based [`crate::policy::Policy`] implements it too, as the cold-start
//! fallback — so a deployment routes sensibly before any eval data exists, and a
//! learned policy can delegate to it when it has no profiles for a pool.

use crate::classify::Request;
use crate::registry::{Level, Modality, Registry};
use serde::{Deserialize, Serialize};

/// Sentinel model id a [`Route`] carries when the safety guard refused the
/// request. The mechanism checks this before dispatch and returns a refusal
/// instead of serving. A `pub const` because both the brain (sets it) and the
/// mechanism (checks it) share the sentinel across the boundary.
pub const REFUSED_MODEL: &str = "refused";

/// Confidence a rule-based cold-start route reports: enough to act on, low
/// enough that a learned policy knows it is an un-personalized guess.
pub const COLD_START_CONFIDENCE: f32 = 0.25;

/// Who the request is for. The id keys per-user adaptation in a learned policy;
/// the rule-based policy ignores it.
#[derive(Debug, Clone)]
pub struct User {
    pub id: String,
}

impl User {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }

    /// An unidentified caller — routed by the global (un-personalized) policy.
    pub fn anonymous() -> Self {
        Self { id: String::new() }
    }
}

/// The per-request service-level objective: hard feasibility ceilings plus the
/// soft cost/latency trade the selector applies. A *value* the caller sets per
/// request (the operator's budget), distinct from a user's learned taste. `0`
/// disables a ceiling.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
pub struct Slo {
    pub max_latency_ms: f32,
    pub max_cost: f32,
    pub min_quality: f32,
    /// Soft weight on (normalized) cost in the selector objective.
    pub lambda_cost: f32,
    /// Soft weight on (normalized) latency in the selector objective.
    pub mu_latency: f32,
}

impl Default for Slo {
    fn default() -> Self {
        Self {
            max_latency_ms: 0.0,
            max_cost: 0.0,
            min_quality: 0.0,
            lambda_cost: 0.3,
            mu_latency: 0.3,
        }
    }
}

/// The routing decision: which model, at what effort level, in what modality,
/// and how confident the policy is. When `model == REFUSED_MODEL` the guard
/// blocked the request and `confidence` is the guard's block confidence.
#[derive(Debug, Clone, PartialEq)]
pub struct Route {
    pub model: String,
    pub level: Level,
    pub modality: Modality,
    pub confidence: f32,
}

impl Route {
    pub fn refused(confidence: f32) -> Self {
        Self {
            model: REFUSED_MODEL.to_string(),
            level: Level::default(),
            modality: Modality::Text,
            confidence,
        }
    }

    pub fn is_refused(&self) -> bool {
        self.model == REFUSED_MODEL
    }
}

/// The seam. Given a request, the user it's for, the SLO, and the pool, decide a
/// [`Route`]. Pure over its inputs.
pub trait RoutePolicy {
    fn route(&self, req: &Request, user: &User, slo: &Slo, registry: &Registry) -> Route;
}
