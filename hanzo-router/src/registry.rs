//! The model pool: what can be served, where it runs, and what it's good at.

use serde::{Deserialize, Serialize};

/// Coarse task buckets used to match a request to a model's strengths. Heuristic
/// today (see [`crate::classify`]); a learned classifier can produce the same
/// value without changing the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Task {
    Code,
    Reasoning,
    Math,
    Creative,
    Vision,
    LongContext,
    CheapChat,
    General,
}

impl Task {
    pub const ALL: [Task; 8] = [
        Task::Code,
        Task::Reasoning,
        Task::Math,
        Task::Creative,
        Task::Vision,
        Task::LongContext,
        Task::CheapChat,
        Task::General,
    ];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&t| t == self).unwrap()
    }
}

/// The I/O pool a request and a model live in. Orthogonal to [`Task`]: `Task` is
/// the text-task taxonomy used to rank quality within a pool; `Modality` selects
/// the pool itself, so the same router serves chat, vision, image-gen and dub by
/// matching pools. Media-generation requests (image/dub) carry [`Task::General`].
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    #[default]
    Text,
    Vision,
    Image,
    Audio,
    Video,
}

impl Modality {
    pub const ALL: [Modality; 5] = [
        Modality::Text,
        Modality::Vision,
        Modality::Image,
        Modality::Audio,
        Modality::Video,
    ];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&m| m == self).unwrap()
    }
}

/// Effort/quality tier a model is served at (reasoning depth, quant, renderer
/// quality). A model can expose several levels; each (model, level, modality) is
/// one profile row the router ranks independently.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[serde(rename_all = "snake_case")]
pub enum Level {
    Fast,
    #[default]
    Balanced,
    Max,
}

impl Level {
    pub const ALL: [Level; 3] = [Level::Fast, Level::Balanced, Level::Max];

    pub fn index(self) -> usize {
        Self::ALL.iter().position(|&l| l == self).unwrap()
    }
}

/// Where a model executes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Backend {
    /// Served by a local hanzo-engine. `est_bytes` is the resident footprint
    /// (quantized weights + overhead) the router fits against available memory.
    Local { est_bytes: u64 },
    /// Served by a cloud provider (OpenAI/Anthropic/...) over the gateway.
    Cloud { provider: String },
}

impl Backend {
    pub fn is_local(&self) -> bool {
        matches!(self, Backend::Local { .. })
    }
}

/// One entry in the pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// Canonical model id (e.g. `deepseek-v4-flash`, `claude-sonnet-4-5`).
    pub id: String,
    pub backend: Backend,
    /// Tasks this model is preferred for (most-preferred first is not required;
    /// ranking is by the policy's task preference order).
    #[serde(default)]
    pub tasks: Vec<Task>,
    /// Max context window (tokens). Used to filter for `LongContext`.
    #[serde(default)]
    pub max_context: usize,
    /// Whether this model accepts image/video input.
    #[serde(default)]
    pub vision: bool,
    /// Relative cost per 1k tokens (any consistent unit); cloud tie-break + ceiling.
    #[serde(default)]
    pub cost_per_1k: f64,
}

impl ModelCard {
    pub fn est_bytes(&self) -> Option<u64> {
        match &self.backend {
            Backend::Local { est_bytes } => Some(*est_bytes),
            Backend::Cloud { .. } => None,
        }
    }
}

/// The full pool the router selects from.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    pub models: Vec<ModelCard>,
}

impl Registry {
    pub fn new(models: Vec<ModelCard>) -> Self {
        Self { models }
    }

    pub fn get(&self, id: &str) -> Option<&ModelCard> {
        self.models.iter().find(|m| m.id == id)
    }

    /// Models that advertise `task` (or `General` as a catch-all).
    pub fn for_task(&self, task: Task) -> impl Iterator<Item = &ModelCard> {
        self.models
            .iter()
            .filter(move |m| m.tasks.contains(&task) || m.tasks.contains(&Task::General))
    }
}
