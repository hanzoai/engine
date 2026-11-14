//! Native active-inference boundary for `/v1/worlds/{world}/inference` hosts.
//!
//! The host authenticates identity, resolves the world and supplies its immutable generative
//! model. Callers can submit beliefs/candidate policies, but cannot overwrite preferences,
//! transition/observation tables or admissibility. There is no actuator interface.
//! The canonical math lives in hanzoai/ml's `hanzo-active-inference`, never copied here.
use hanzo_active_inference::{Model, Policies, Posterior, Result};
use serde::{Deserialize, Serialize};

/// A request to infer a state and rank explicit policies.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// Current prior in the world's registered latent state space.
    pub belief: Vec<f64>,
    /// Optional current observation: (likelihood action, outcome).
    pub observation: Option<(usize, usize)>,
    /// Candidate open-loop action sequences; the host assigns policy admissibility.
    pub policies: Vec<Vec<usize>>,
}
/// Mathematical results with no execution authorization.
#[derive(Debug, Clone, Serialize)]
pub struct Response {
    /// Filtered current state, if an observation was supplied.
    pub posterior: Option<Posterior>,
    /// Exact finite-state expected free-energy terms and policy probabilities.
    pub policies: Policies,
    /// This inference result never authorizes external execution.
    pub execution_authorized: bool,
}
/// Immutable model and host-owned precision. An instance belongs to one org/world revision.
pub struct WorldInference {
    model: Model,
    precision: f64,
}
impl WorldInference {
    /// Load a validated generative model. Precision is validated at inference.
    pub fn new(model: Model, precision: f64) -> Self {
        Self { model, precision }
    }
    /// Update beliefs, then infer a policy posterior under host-owned habits and safety mask.
    pub fn infer(
        &self,
        request: &Request,
        habits: &[f64],
        admissible: &[bool],
    ) -> Result<Response> {
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
        Ok(Response {
            posterior,
            policies,
            execution_authorized: false,
        })
    }
}
/// Construct a finite-state model without coupling consumers to a tensor backend.
pub use hanzo_active_inference::Model as GenerativeModel;
/// Re-export the one canonical model and learning types for in-process engine hosts.
pub use hanzo_active_inference::{Counts, Error, Terms};
