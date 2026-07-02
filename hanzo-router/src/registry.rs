//! The model pool: what can be served, where it runs, and what it's good at.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Public route families. These stay intentionally coarse and product-shaped:
/// callers can ask for a route family without needing to know which specialist
/// lane or provider graph is used underneath.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PublicRoute {
    #[serde(rename = "zen-flash")]
    ZenFlash,
    #[serde(rename = "zen-normal")]
    ZenNormal,
    #[serde(rename = "zen-pro")]
    ZenPro,
    #[serde(rename = "zen-max")]
    ZenMax,
    #[serde(rename = "zen-code")]
    ZenCode,
    #[serde(rename = "zen-agent")]
    ZenAgent,
    #[serde(rename = "zen-local")]
    ZenLocal,
    #[serde(rename = "zen-auto+")]
    ZenAutoPlus,
}

impl PublicRoute {
    pub fn as_str(self) -> &'static str {
        match self {
            PublicRoute::ZenFlash => "zen-flash",
            PublicRoute::ZenNormal => "zen-normal",
            PublicRoute::ZenPro => "zen-pro",
            PublicRoute::ZenMax => "zen-max",
            PublicRoute::ZenCode => "zen-code",
            PublicRoute::ZenAgent => "zen-agent",
            PublicRoute::ZenLocal => "zen-local",
            PublicRoute::ZenAutoPlus => "zen-auto+",
        }
    }
}

/// Internal specialist lanes. These are not required to be public model names;
/// they are the lanes the router uses to build a specialist graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SpecialistLane {
    #[serde(rename = "zen-text")]
    Text,
    #[serde(rename = "zen-reasoning")]
    Reasoning,
    #[serde(rename = "zen-code")]
    Code,
    #[serde(rename = "zen-agent")]
    Agent,
    #[serde(rename = "zen-local")]
    Local,
    #[serde(rename = "zen-image")]
    Image,
    #[serde(rename = "zen-design")]
    Design,
    #[serde(rename = "zen-product")]
    Product,
    #[serde(rename = "zen-video")]
    Video,
    #[serde(rename = "zen-vision")]
    Vision,
    #[serde(rename = "zen-ocr")]
    Ocr,
    #[serde(rename = "zen-omni")]
    Omni,
    #[serde(rename = "zen-rag")]
    Rag,
    #[serde(rename = "zen-rerank")]
    Rerank,
    #[serde(rename = "zen-embed")]
    Embed,
    #[serde(rename = "zen-speech")]
    Speech,
    #[serde(rename = "zen-voice")]
    Voice,
    #[serde(rename = "zen-music")]
    Music,
    #[serde(rename = "zen-safety")]
    Safety,
    #[serde(rename = "zen-judge")]
    Judge,
}

impl SpecialistLane {
    pub fn as_str(self) -> &'static str {
        match self {
            SpecialistLane::Text => "zen-text",
            SpecialistLane::Reasoning => "zen-reasoning",
            SpecialistLane::Code => "zen-code",
            SpecialistLane::Agent => "zen-agent",
            SpecialistLane::Local => "zen-local",
            SpecialistLane::Image => "zen-image",
            SpecialistLane::Design => "zen-design",
            SpecialistLane::Product => "zen-product",
            SpecialistLane::Video => "zen-video",
            SpecialistLane::Vision => "zen-vision",
            SpecialistLane::Ocr => "zen-ocr",
            SpecialistLane::Omni => "zen-omni",
            SpecialistLane::Rag => "zen-rag",
            SpecialistLane::Rerank => "zen-rerank",
            SpecialistLane::Embed => "zen-embed",
            SpecialistLane::Speech => "zen-speech",
            SpecialistLane::Voice => "zen-voice",
            SpecialistLane::Music => "zen-music",
            SpecialistLane::Safety => "zen-safety",
            SpecialistLane::Judge => "zen-judge",
        }
    }

    pub fn default_purpose(self) -> &'static str {
        match self {
            SpecialistLane::Text => "synthesize the text answer",
            SpecialistLane::Reasoning => "perform deep reasoning",
            SpecialistLane::Code => "draft or repair code",
            SpecialistLane::Agent => "coordinate tool, browser, or computer actions",
            SpecialistLane::Local => "prefer local execution",
            SpecialistLane::Image => "generate or edit images",
            SpecialistLane::Design => "generate typography, layout, brand, or ad assets",
            SpecialistLane::Product => "generate product, ecommerce, catalog, or fashion visuals",
            SpecialistLane::Video => "generate or animate video",
            SpecialistLane::Vision => "understand images, screenshots, diagrams, or charts",
            SpecialistLane::Ocr => "extract document text, tables, forms, or layout",
            SpecialistLane::Omni => "handle mixed image, video, audio, and text input",
            SpecialistLane::Rag => "retrieve, compress, and answer from sources",
            SpecialistLane::Rerank => "rerank evidence for precision",
            SpecialistLane::Embed => "embed content for search, memory, or retrieval",
            SpecialistLane::Speech => "transcribe or understand speech",
            SpecialistLane::Voice => "generate speech or voice",
            SpecialistLane::Music => "generate music or sound effects",
            SpecialistLane::Safety => "moderate content and check risky actions",
            SpecialistLane::Judge => "verify, grade, or compare outputs",
        }
    }
}

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
    Ocr,
    Image,
    Design,
    Product,
    Video,
    LongContext,
    Rag,
    Embed,
    Rerank,
    Speech,
    Voice,
    Music,
    Safety,
    Judge,
    Agent,
    CheapChat,
    General,
}

impl Task {
    pub fn specialist_lane(self) -> SpecialistLane {
        match self {
            Task::Code => SpecialistLane::Code,
            Task::Reasoning | Task::Math => SpecialistLane::Reasoning,
            Task::Creative | Task::Image => SpecialistLane::Image,
            Task::Design => SpecialistLane::Design,
            Task::Product => SpecialistLane::Product,
            Task::Video => SpecialistLane::Video,
            Task::Vision => SpecialistLane::Vision,
            Task::Ocr => SpecialistLane::Ocr,
            Task::LongContext | Task::Rag => SpecialistLane::Rag,
            Task::Embed => SpecialistLane::Embed,
            Task::Rerank => SpecialistLane::Rerank,
            Task::Speech => SpecialistLane::Speech,
            Task::Voice => SpecialistLane::Voice,
            Task::Music => SpecialistLane::Music,
            Task::Safety => SpecialistLane::Safety,
            Task::Judge => SpecialistLane::Judge,
            Task::Agent => SpecialistLane::Agent,
            Task::CheapChat | Task::General => SpecialistLane::Text,
        }
    }
}

/// Modality metadata for commercial routing and provider selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    #[default]
    Text,
    Vision,
    Image,
    Video,
    Audio,
    Embedding,
    Rerank,
    Safety,
    Judge,
    Music,
    Multimodal,
    Other,
}

/// Whether the artifact is closed, open-weight, open-source, or local-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum OpenOrClosed {
    #[default]
    Closed,
    OpenWeight,
    OpenSource,
    Local,
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

impl Default for Backend {
    fn default() -> Self {
        Backend::Cloud {
            provider: String::new(),
        }
    }
}

fn default_commercial_resale_allowed() -> bool {
    true
}

/// Numeric capability profile. Scores are normalized to 0.0-1.0 and represent
/// "goodness" for the lane: high `latency` means low latency / fast response.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapabilityProfile {
    #[serde(default)]
    pub text_generation: f64,
    #[serde(default)]
    pub reasoning_depth: f64,
    #[serde(default)]
    pub coding_repair: f64,
    #[serde(default)]
    pub terminal_agent: f64,
    #[serde(default)]
    pub browser_agent: f64,
    #[serde(default)]
    pub tool_calling: f64,
    #[serde(default)]
    pub long_context_retrieval: f64,
    #[serde(default)]
    pub long_context_transformation: f64,
    #[serde(default)]
    pub vision_understanding: f64,
    #[serde(default)]
    pub ocr_layout: f64,
    #[serde(default)]
    pub image_generation: f64,
    #[serde(default)]
    pub typography_generation: f64,
    #[serde(default)]
    pub video_generation: f64,
    #[serde(default)]
    pub speech_to_text: f64,
    #[serde(default)]
    pub text_to_speech: f64,
    #[serde(default)]
    pub music_generation: f64,
    #[serde(default)]
    pub embedding_quality: f64,
    #[serde(default)]
    pub rerank_quality: f64,
    #[serde(default)]
    pub safety_filter: f64,
    #[serde(default)]
    pub judge_quality: f64,
    #[serde(default)]
    pub latency: f64,
    #[serde(default)]
    pub throughput: f64,
    #[serde(default)]
    pub cost_efficiency: f64,
    #[serde(default)]
    pub local_deployability: f64,
    #[serde(default)]
    pub brand_design: f64,
    #[serde(default = "default_commercial_resale_allowed")]
    pub commercial_license_ok: bool,
}

impl Default for CapabilityProfile {
    fn default() -> Self {
        Self {
            text_generation: 0.0,
            reasoning_depth: 0.0,
            coding_repair: 0.0,
            terminal_agent: 0.0,
            browser_agent: 0.0,
            tool_calling: 0.0,
            long_context_retrieval: 0.0,
            long_context_transformation: 0.0,
            vision_understanding: 0.0,
            ocr_layout: 0.0,
            image_generation: 0.0,
            typography_generation: 0.0,
            video_generation: 0.0,
            speech_to_text: 0.0,
            text_to_speech: 0.0,
            music_generation: 0.0,
            embedding_quality: 0.0,
            rerank_quality: 0.0,
            safety_filter: 0.0,
            judge_quality: 0.0,
            latency: 0.0,
            throughput: 0.0,
            cost_efficiency: 0.0,
            local_deployability: 0.0,
            brand_design: 0.0,
            commercial_license_ok: true,
        }
    }
}

impl CapabilityProfile {
    pub fn score_against(&self, demand: &CapabilityProfile) -> f64 {
        self.text_generation * demand.text_generation
            + self.reasoning_depth * demand.reasoning_depth
            + self.coding_repair * demand.coding_repair
            + self.terminal_agent * demand.terminal_agent
            + self.browser_agent * demand.browser_agent
            + self.tool_calling * demand.tool_calling
            + self.long_context_retrieval * demand.long_context_retrieval
            + self.long_context_transformation * demand.long_context_transformation
            + self.vision_understanding * demand.vision_understanding
            + self.ocr_layout * demand.ocr_layout
            + self.image_generation * demand.image_generation
            + self.typography_generation * demand.typography_generation
            + self.video_generation * demand.video_generation
            + self.speech_to_text * demand.speech_to_text
            + self.text_to_speech * demand.text_to_speech
            + self.music_generation * demand.music_generation
            + self.embedding_quality * demand.embedding_quality
            + self.rerank_quality * demand.rerank_quality
            + self.safety_filter * demand.safety_filter
            + self.judge_quality * demand.judge_quality
            + self.latency * demand.latency
            + self.throughput * demand.throughput
            + self.cost_efficiency * demand.cost_efficiency
            + self.local_deployability * demand.local_deployability
            + self.brand_design * demand.brand_design
    }

    pub fn for_lane(lane: SpecialistLane) -> Self {
        match lane {
            SpecialistLane::Text => Self {
                text_generation: 1.0,
                cost_efficiency: 0.3,
                latency: 0.3,
                ..Self::default()
            },
            SpecialistLane::Reasoning => Self {
                text_generation: 0.5,
                reasoning_depth: 1.0,
                judge_quality: 0.2,
                ..Self::default()
            },
            SpecialistLane::Code => Self {
                coding_repair: 1.0,
                terminal_agent: 0.5,
                reasoning_depth: 0.5,
                tool_calling: 0.4,
                ..Self::default()
            },
            SpecialistLane::Agent => Self {
                browser_agent: 1.0,
                terminal_agent: 0.7,
                tool_calling: 1.0,
                reasoning_depth: 0.5,
                ..Self::default()
            },
            SpecialistLane::Local => Self {
                local_deployability: 1.0,
                cost_efficiency: 0.7,
                ..Self::default()
            },
            SpecialistLane::Image => Self {
                image_generation: 1.0,
                vision_understanding: 0.2,
                ..Self::default()
            },
            SpecialistLane::Design => Self {
                image_generation: 0.6,
                typography_generation: 1.0,
                brand_design: 0.9,
                ..Self::default()
            },
            SpecialistLane::Product => Self {
                image_generation: 0.7,
                typography_generation: 0.5,
                brand_design: 0.8,
                vision_understanding: 0.5,
                ..Self::default()
            },
            SpecialistLane::Video => Self {
                video_generation: 1.0,
                image_generation: 0.2,
                ..Self::default()
            },
            SpecialistLane::Vision => Self {
                vision_understanding: 1.0,
                reasoning_depth: 0.3,
                ..Self::default()
            },
            SpecialistLane::Ocr => Self {
                ocr_layout: 1.0,
                vision_understanding: 0.5,
                long_context_retrieval: 0.2,
                ..Self::default()
            },
            SpecialistLane::Omni => Self {
                vision_understanding: 0.8,
                speech_to_text: 0.5,
                text_generation: 0.5,
                tool_calling: 0.3,
                ..Self::default()
            },
            SpecialistLane::Rag => Self {
                long_context_retrieval: 1.0,
                long_context_transformation: 0.7,
                text_generation: 0.4,
                ..Self::default()
            },
            SpecialistLane::Rerank => Self {
                rerank_quality: 1.0,
                long_context_retrieval: 0.5,
                ..Self::default()
            },
            SpecialistLane::Embed => Self {
                embedding_quality: 1.0,
                throughput: 0.5,
                cost_efficiency: 0.4,
                ..Self::default()
            },
            SpecialistLane::Speech => Self {
                speech_to_text: 1.0,
                latency: 0.4,
                ..Self::default()
            },
            SpecialistLane::Voice => Self {
                text_to_speech: 1.0,
                latency: 0.3,
                ..Self::default()
            },
            SpecialistLane::Music => Self {
                music_generation: 1.0,
                ..Self::default()
            },
            SpecialistLane::Safety => Self {
                safety_filter: 1.0,
                latency: 0.4,
                ..Self::default()
            },
            SpecialistLane::Judge => Self {
                judge_quality: 1.0,
                reasoning_depth: 0.5,
                ..Self::default()
            },
        }
    }
}

/// One entry in the pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelCard {
    /// Canonical model id (e.g. `deepseek-v4-flash`, `claude-sonnet-4-5`).
    pub id: String,
    /// Commercial/provider-facing provider name. If omitted, cloud backends use
    /// the provider embedded in [`Backend::Cloud`].
    #[serde(default)]
    pub provider: String,
    /// Provider model id. Defaults to `id` when empty.
    #[serde(default)]
    pub model: String,
    pub backend: Backend,
    /// Tasks this model is preferred for (most-preferred first is not required;
    /// ranking is by the policy's task preference order).
    #[serde(default)]
    pub tasks: Vec<Task>,
    /// Internal specialist lanes this model can serve.
    #[serde(default)]
    pub lanes: Vec<SpecialistLane>,
    /// Max context window (tokens). Used to filter for `LongContext`.
    #[serde(default)]
    pub max_context: usize,
    /// Whether this model accepts image/video input.
    #[serde(default)]
    pub vision: bool,
    /// Relative cost per 1k tokens (any consistent unit); cloud tie-break + ceiling.
    #[serde(default)]
    pub cost_per_1k: f64,
    #[serde(default)]
    pub modality: Modality,
    #[serde(default)]
    pub open_or_closed: OpenOrClosed,
    #[serde(default)]
    pub weights_license: String,
    #[serde(default = "default_commercial_resale_allowed")]
    pub commercial_resale_allowed: bool,
    #[serde(default)]
    pub data_retention: String,
    #[serde(default)]
    pub hipaa_baa: bool,
    #[serde(default)]
    pub soc2: bool,
    #[serde(default)]
    pub region: Vec<String>,
    #[serde(default)]
    pub price_input: f64,
    #[serde(default)]
    pub price_output: f64,
    #[serde(default)]
    pub price_per_image: f64,
    #[serde(default)]
    pub price_per_video_second: f64,
    #[serde(default)]
    pub latency_p50: f64,
    #[serde(default)]
    pub latency_p95: f64,
    #[serde(default)]
    pub quality_score_by_lane: BTreeMap<SpecialistLane, f64>,
    #[serde(default)]
    pub capabilities: CapabilityProfile,
}

impl ModelCard {
    pub fn est_bytes(&self) -> Option<u64> {
        match &self.backend {
            Backend::Local { est_bytes } => Some(*est_bytes),
            Backend::Cloud { .. } => None,
        }
    }

    pub fn provider_name(&self) -> &str {
        if !self.provider.is_empty() {
            &self.provider
        } else {
            match &self.backend {
                Backend::Local { .. } => "local",
                Backend::Cloud { provider } => provider,
            }
        }
    }

    pub fn model_name(&self) -> &str {
        if self.model.is_empty() {
            &self.id
        } else {
            &self.model
        }
    }

    pub fn lane_score(&self, lane: SpecialistLane) -> f64 {
        self.quality_score_by_lane
            .get(&lane)
            .copied()
            .unwrap_or_default()
    }
}

impl Default for ModelCard {
    fn default() -> Self {
        Self {
            id: String::new(),
            provider: String::new(),
            model: String::new(),
            backend: Backend::default(),
            tasks: Vec::new(),
            lanes: Vec::new(),
            max_context: 0,
            vision: false,
            cost_per_1k: 0.0,
            modality: Modality::default(),
            open_or_closed: OpenOrClosed::default(),
            weights_license: String::new(),
            commercial_resale_allowed: true,
            data_retention: String::new(),
            hipaa_baa: false,
            soc2: false,
            region: Vec::new(),
            price_input: 0.0,
            price_output: 0.0,
            price_per_image: 0.0,
            price_per_video_second: 0.0,
            latency_p50: 0.0,
            latency_p95: 0.0,
            quality_score_by_lane: BTreeMap::new(),
            capabilities: CapabilityProfile::default(),
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

    /// Models that advertise an internal specialist lane.
    pub fn for_lane(&self, lane: SpecialistLane) -> impl Iterator<Item = &ModelCard> {
        self.models.iter().filter(move |m| m.lanes.contains(&lane))
    }

    /// A curated, disparate-model-native seed registry. Prices and live latency
    /// are zero placeholders until the gateway imports provider pricing; resale
    /// filtering should be enforced through `commercial_resale_allowed`.
    pub fn disparate_sota() -> Self {
        let mut models = Vec::new();
        add_frontier_text(&mut models);
        add_code_and_agents(&mut models);
        add_image_design(&mut models);
        add_video(&mut models);
        add_vision_ocr(&mut models);
        add_rag(&mut models);
        add_speech_voice_music(&mut models);
        add_safety_and_judges(&mut models);
        Self::new(models)
    }
}

fn lane_scores(scores: &[(SpecialistLane, f64)]) -> BTreeMap<SpecialistLane, f64> {
    scores.iter().copied().collect()
}

fn closed_cloud(
    id: &str,
    provider: &str,
    model: &str,
    modality: Modality,
    tasks: Vec<Task>,
    lanes: Vec<SpecialistLane>,
    capabilities: CapabilityProfile,
    scores: &[(SpecialistLane, f64)],
) -> ModelCard {
    ModelCard {
        id: id.into(),
        provider: provider.into(),
        model: model.into(),
        backend: Backend::Cloud {
            provider: provider.into(),
        },
        tasks,
        lanes,
        modality,
        open_or_closed: OpenOrClosed::Closed,
        weights_license: "closed/api".into(),
        commercial_resale_allowed: true,
        data_retention: "provider_policy".into(),
        soc2: true,
        region: vec!["us".into(), "eu".into()],
        quality_score_by_lane: lane_scores(scores),
        capabilities,
        ..ModelCard::default()
    }
}

fn open_weight(
    id: &str,
    provider: &str,
    model: &str,
    modality: Modality,
    tasks: Vec<Task>,
    lanes: Vec<SpecialistLane>,
    capabilities: CapabilityProfile,
    scores: &[(SpecialistLane, f64)],
) -> ModelCard {
    ModelCard {
        id: id.into(),
        provider: provider.into(),
        model: model.into(),
        backend: Backend::Cloud {
            provider: provider.into(),
        },
        tasks,
        lanes,
        modality,
        open_or_closed: OpenOrClosed::OpenWeight,
        weights_license: "verify_before_resale".into(),
        commercial_resale_allowed: false,
        data_retention: "self_host_or_provider_policy".into(),
        region: vec!["us".into(), "eu".into()],
        quality_score_by_lane: lane_scores(scores),
        capabilities: CapabilityProfile {
            commercial_license_ok: false,
            ..capabilities
        },
        ..ModelCard::default()
    }
}

fn local_open(
    id: &str,
    est_bytes: u64,
    modality: Modality,
    tasks: Vec<Task>,
    lanes: Vec<SpecialistLane>,
    capabilities: CapabilityProfile,
    scores: &[(SpecialistLane, f64)],
) -> ModelCard {
    ModelCard {
        id: id.into(),
        provider: "local".into(),
        model: id.into(),
        backend: Backend::Local { est_bytes },
        tasks,
        lanes,
        modality,
        open_or_closed: OpenOrClosed::Local,
        weights_license: "local_deployment".into(),
        commercial_resale_allowed: true,
        data_retention: "local".into(),
        region: vec!["local".into()],
        quality_score_by_lane: lane_scores(scores),
        capabilities,
        ..ModelCard::default()
    }
}

fn add_frontier_text(models: &mut Vec<ModelCard>) {
    let lanes = vec![SpecialistLane::Text, SpecialistLane::Reasoning];
    let tasks = vec![Task::General, Task::Reasoning, Task::Math, Task::LongContext];
    for (id, provider, model, score) in [
        ("gpt-5.5", "openai", "gpt-5.5", 0.99),
        ("claude-opus-4.8", "anthropic", "claude-opus-4.8", 0.98),
        ("gemini-3.1-pro", "google", "gemini-3.1-pro", 0.97),
        ("grok-frontier", "xai", "grok-frontier", 0.94),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Text,
            tasks.clone(),
            lanes.clone(),
            CapabilityProfile {
                text_generation: score,
                reasoning_depth: score,
                long_context_transformation: 0.9,
                tool_calling: 0.8,
                judge_quality: 0.7,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Text, score), (SpecialistLane::Reasoning, score)],
        ));
    }

    for (id, provider, model, score) in [
        ("glm-5.2", "zai", "glm-5.2", 0.94),
        ("deepseek-v4-pro-max", "deepseek", "deepseek-v4-pro-max", 0.95),
        ("deepseek-v4-pro-high", "deepseek", "deepseek-v4-pro-high", 0.93),
        ("minimax-m3", "minimax", "minimax-m3", 0.92),
        ("kimi-k2.7-thinking", "moonshot", "kimi-k2.7-thinking", 0.93),
        ("kimi-k2.6", "moonshot", "kimi-k2.6", 0.90),
        ("qwen3.5-397b-reasoning", "qwen", "qwen3.5-397b-reasoning", 0.92),
        ("nemotron-3-ultra", "nvidia", "nemotron-3-ultra", 0.91),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Text,
            tasks.clone(),
            lanes.clone(),
            CapabilityProfile {
                text_generation: score,
                reasoning_depth: score,
                long_context_transformation: 0.85,
                tool_calling: 0.7,
                cost_efficiency: 0.75,
                local_deployability: 0.4,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Text, score), (SpecialistLane::Reasoning, score)],
        ));
    }
}

fn add_code_and_agents(models: &mut Vec<ModelCard>) {
    let code_tasks = vec![Task::Code, Task::Reasoning, Task::General];
    let agent_tasks = vec![Task::Agent, Task::Code, Task::General];
    for (id, provider, model, code_score, agent_score) in [
        ("gpt-5.5-code", "openai", "gpt-5.5", 0.99, 0.95),
        ("claude-opus-4.8-agent", "anthropic", "claude-opus-4.8", 0.98, 0.99),
        ("gemini-3.1-pro-code", "google", "gemini-3.1-pro", 0.95, 0.93),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Text,
            if agent_score > code_score {
                agent_tasks.clone()
            } else {
                code_tasks.clone()
            },
            vec![SpecialistLane::Code, SpecialistLane::Agent, SpecialistLane::Judge],
            CapabilityProfile {
                coding_repair: code_score,
                terminal_agent: agent_score,
                browser_agent: agent_score,
                tool_calling: 0.95,
                reasoning_depth: 0.9,
                judge_quality: 0.8,
                ..CapabilityProfile::default()
            },
            &[
                (SpecialistLane::Code, code_score),
                (SpecialistLane::Agent, agent_score),
                (SpecialistLane::Judge, 0.82),
            ],
        ));
    }

    for (id, provider, model, code_score, agent_score) in [
        ("deepseek-v4-pro-max-code", "deepseek", "deepseek-v4-pro-max", 0.97, 0.88),
        ("glm-5.2-code", "zai", "glm-5.2", 0.94, 0.9),
        ("kimi-k2.7-code", "moonshot", "kimi-k2.7-code", 0.95, 0.9),
        ("minimax-m3-agent", "minimax", "minimax-m3", 0.91, 0.93),
        ("qwen3.5-397b-code", "qwen", "qwen3.5-397b-reasoning", 0.92, 0.84),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Text,
            code_tasks.clone(),
            vec![SpecialistLane::Code, SpecialistLane::Agent],
            CapabilityProfile {
                coding_repair: code_score,
                terminal_agent: agent_score,
                browser_agent: agent_score,
                tool_calling: 0.75,
                reasoning_depth: 0.85,
                cost_efficiency: 0.8,
                local_deployability: 0.5,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Code, code_score), (SpecialistLane::Agent, agent_score)],
        ));
    }

    models.push(local_open(
        "zen5-coder-local",
        93 << 30,
        Modality::Text,
        vec![Task::Code, Task::General],
        vec![SpecialistLane::Code, SpecialistLane::Local],
        CapabilityProfile {
            coding_repair: 0.9,
            terminal_agent: 0.75,
            local_deployability: 1.0,
            cost_efficiency: 1.0,
            ..CapabilityProfile::default()
        },
        &[(SpecialistLane::Code, 0.9), (SpecialistLane::Local, 1.0)],
    ));
}

fn add_image_design(models: &mut Vec<ModelCard>) {
    for (id, provider, model, image_score, design_score) in [
        ("gpt-image", "openai", "gpt-image", 0.94, 0.9),
        ("gemini-image", "google", "gemini-image", 0.92, 0.88),
        ("midjourney", "midjourney", "midjourney", 0.96, 0.88),
        ("ideogram-commercial", "ideogram", "ideogram-api", 0.93, 0.98),
        ("runway-image", "runway", "runway-image", 0.9, 0.85),
        ("adobe-firefly", "adobe", "firefly", 0.9, 0.92),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Image,
            vec![Task::Image, Task::Design, Task::Product, Task::Creative],
            vec![SpecialistLane::Image, SpecialistLane::Design, SpecialistLane::Product],
            CapabilityProfile {
                image_generation: image_score,
                typography_generation: design_score,
                brand_design: design_score,
                cost_efficiency: 0.4,
                ..CapabilityProfile::default()
            },
            &[
                (SpecialistLane::Image, image_score),
                (SpecialistLane::Design, design_score),
                (SpecialistLane::Product, (image_score + design_score) / 2.0),
            ],
        ));
    }

    for (id, provider, model, image_score, design_score) in [
        ("ideogram-4", "ideogram", "ideogram-4", 0.92, 0.97),
        ("flux.2-dev", "black-forest-labs", "flux.2-dev", 0.93, 0.82),
        ("flux-family", "black-forest-labs", "flux", 0.9, 0.8),
        ("qwen-image", "qwen", "qwen-image", 0.9, 0.85),
        ("hunyuanimage-3.0", "tencent", "hunyuanimage-3.0", 0.89, 0.8),
        ("sdxl-finetunes", "stability", "sdxl-finetunes", 0.82, 0.72),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Image,
            vec![Task::Image, Task::Design, Task::Product, Task::Creative],
            vec![SpecialistLane::Image, SpecialistLane::Design, SpecialistLane::Product],
            CapabilityProfile {
                image_generation: image_score,
                typography_generation: design_score,
                brand_design: design_score,
                local_deployability: 0.8,
                cost_efficiency: 0.9,
                ..CapabilityProfile::default()
            },
            &[
                (SpecialistLane::Image, image_score),
                (SpecialistLane::Design, design_score),
                (SpecialistLane::Product, (image_score + design_score) / 2.0),
            ],
        ));
    }
}

fn add_video(models: &mut Vec<ModelCard>) {
    for (id, provider, model, score) in [
        ("veo", "google", "veo", 0.97),
        ("sora", "openai", "sora", 0.96),
        ("runway-gen", "runway", "runway-gen", 0.94),
        ("kling", "kuaishou", "kling", 0.93),
        ("pika", "pika", "pika", 0.86),
        ("luma-dream-machine", "luma", "dream-machine", 0.9),
        ("minimax-video", "minimax", "minimax-video", 0.88),
        ("hailuo-video", "minimax", "hailuo", 0.88),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Video,
            vec![Task::Video, Task::Creative],
            vec![SpecialistLane::Video],
            CapabilityProfile {
                video_generation: score,
                image_generation: 0.4,
                cost_efficiency: 0.4,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Video, score)],
        ));
    }

    for (id, provider, model, score) in [
        ("wan2.x-video", "alibaba", "wan2.x", 0.88),
        ("hunyuanvideo", "tencent", "hunyuanvideo", 0.87),
        ("cogvideox", "zhipu", "cogvideox", 0.84),
        ("ltx-video", "lightricks", "ltx-video", 0.8),
        ("open-sora-style", "community", "open-sora-style", 0.76),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Video,
            vec![Task::Video, Task::Creative],
            vec![SpecialistLane::Video],
            CapabilityProfile {
                video_generation: score,
                local_deployability: 0.7,
                cost_efficiency: 0.8,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Video, score)],
        ));
    }
}

fn add_vision_ocr(models: &mut Vec<ModelCard>) {
    for (id, provider, model, vision_score, ocr_score) in [
        ("gpt-5.5-vision", "openai", "gpt-5.5", 0.97, 0.9),
        ("gemini-3.1-pro-vision", "google", "gemini-3.1-pro", 0.98, 0.92),
        ("claude-opus-4.8-vision", "anthropic", "claude-opus-4.8", 0.95, 0.88),
        ("azure-document-intelligence", "azure", "document-intelligence", 0.7, 0.96),
        ("google-document-ai", "google", "document-ai", 0.72, 0.97),
        ("aws-textract", "aws", "textract", 0.65, 0.9),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Vision,
            vec![Task::Vision, Task::Ocr, Task::General],
            vec![SpecialistLane::Vision, SpecialistLane::Ocr, SpecialistLane::Omni],
            CapabilityProfile {
                vision_understanding: vision_score,
                ocr_layout: ocr_score,
                reasoning_depth: 0.6,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Vision, vision_score), (SpecialistLane::Ocr, ocr_score)],
        ));
    }

    for (id, provider, model, vision_score, ocr_score) in [
        ("minimax-m3-vision", "minimax", "minimax-m3", 0.91, 0.82),
        ("qwen3.5-omni", "qwen", "qwen3.5-omni", 0.92, 0.86),
        ("qwen3-vl", "qwen", "qwen3-vl", 0.9, 0.88),
        ("llama-4-maverick", "meta", "llama-4-maverick", 0.86, 0.72),
        ("llama-4-scout", "meta", "llama-4-scout", 0.82, 0.68),
        ("internvl", "opengvlab", "internvl", 0.86, 0.78),
        ("molmo", "allenai", "molmo", 0.84, 0.72),
        ("varco-vision-2.0", "naver", "varco-vision-2.0", 0.85, 0.9),
        ("paddleocr-vl", "paddlepaddle", "paddleocr-vl", 0.76, 0.96),
        ("mineru", "opendatalab", "mineru", 0.5, 0.88),
        ("glm-ocr", "zai", "glm-ocr", 0.68, 0.9),
        ("tesseract", "tesseract", "tesseract", 0.2, 0.45),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Vision,
            vec![Task::Vision, Task::Ocr, Task::General],
            vec![SpecialistLane::Vision, SpecialistLane::Ocr, SpecialistLane::Omni],
            CapabilityProfile {
                vision_understanding: vision_score,
                ocr_layout: ocr_score,
                local_deployability: 0.8,
                cost_efficiency: 0.8,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Vision, vision_score), (SpecialistLane::Ocr, ocr_score)],
        ));
    }
}

fn add_rag(models: &mut Vec<ModelCard>) {
    for (id, provider, model, embed_score) in [
        ("openai-text-embedding-3", "openai", "text-embedding-3", 0.9),
        ("gemini-embedding", "google", "gemini-embedding", 0.89),
        ("cohere-embed", "cohere", "embed", 0.9),
        ("voyage-embedding", "voyage", "voyage-embedding", 0.92),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Embedding,
            vec![Task::Embed, Task::Rag],
            vec![SpecialistLane::Embed, SpecialistLane::Rag],
            CapabilityProfile {
                embedding_quality: embed_score,
                long_context_retrieval: 0.7,
                throughput: 0.7,
                cost_efficiency: 0.6,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Embed, embed_score), (SpecialistLane::Rag, 0.75)],
        ));
    }

    for (id, provider, model, embed_score) in [
        ("jina-embedding", "jina", "jina-embedding", 0.88),
        ("bge-embedding", "baai", "bge", 0.86),
        ("qwen3-embedding", "qwen", "qwen3-embedding", 0.9),
        ("e5-multilingual", "intfloat", "multilingual-e5", 0.84),
        ("qwen3-vl-embedding", "qwen", "qwen3-vl-embedding", 0.91),
        ("nemotron-colembed-v2", "nvidia", "nemotron-colembed-v2", 0.93),
        ("colpali", "illuin", "colpali", 0.88),
        ("colqwen", "qwen", "colqwen", 0.89),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Embedding,
            vec![Task::Embed, Task::Rag],
            vec![SpecialistLane::Embed, SpecialistLane::Rag, SpecialistLane::Vision],
            CapabilityProfile {
                embedding_quality: embed_score,
                long_context_retrieval: 0.8,
                vision_understanding: 0.4,
                local_deployability: 0.8,
                cost_efficiency: 0.9,
                ..CapabilityProfile::default()
            },
            &[
                (SpecialistLane::Embed, embed_score),
                (SpecialistLane::Rag, 0.82),
                (SpecialistLane::Vision, 0.55),
            ],
        ));
    }

    for (id, provider, model, score, closed) in [
        ("cohere-rerank", "cohere", "rerank", 0.93, true),
        ("voyage-rerank", "voyage", "voyage-rerank", 0.92, true),
        ("jina-reranker", "jina", "jina-reranker", 0.9, false),
        ("bge-reranker", "baai", "bge-reranker", 0.88, false),
        ("qwen3-reranker", "qwen", "qwen3-reranker", 0.91, false),
        ("qwen3-vl-reranker", "qwen", "qwen3-vl-reranker", 0.92, false),
        ("colbert", "stanford", "colbert", 0.87, false),
    ] {
        let card = if closed {
            closed_cloud(
                id,
                provider,
                model,
                Modality::Rerank,
                vec![Task::Rerank, Task::Rag],
                vec![SpecialistLane::Rerank, SpecialistLane::Rag],
                CapabilityProfile {
                    rerank_quality: score,
                    long_context_retrieval: 0.7,
                    ..CapabilityProfile::default()
                },
                &[(SpecialistLane::Rerank, score), (SpecialistLane::Rag, 0.8)],
            )
        } else {
            open_weight(
                id,
                provider,
                model,
                Modality::Rerank,
                vec![Task::Rerank, Task::Rag],
                vec![SpecialistLane::Rerank, SpecialistLane::Rag],
                CapabilityProfile {
                    rerank_quality: score,
                    long_context_retrieval: 0.7,
                    cost_efficiency: 0.85,
                    ..CapabilityProfile::default()
                },
                &[(SpecialistLane::Rerank, score), (SpecialistLane::Rag, 0.8)],
            )
        };
        models.push(card);
    }
}

fn add_speech_voice_music(models: &mut Vec<ModelCard>) {
    for (id, provider, model, score) in [
        ("openai-audio-asr", "openai", "audio-transcription", 0.93),
        ("google-speech-to-text", "google", "speech-to-text", 0.92),
        ("deepgram", "deepgram", "nova", 0.91),
        ("assemblyai", "assemblyai", "universal", 0.9),
        ("elevenlabs-scribe", "elevenlabs", "scribe", 0.9),
        ("speechmatics", "speechmatics", "speechmatics", 0.89),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Audio,
            vec![Task::Speech],
            vec![SpecialistLane::Speech],
            CapabilityProfile {
                speech_to_text: score,
                latency: 0.7,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Speech, score)],
        ));
    }

    for (id, provider, model, score) in [
        ("whisper-large-v3", "openai", "whisper-large-v3", 0.86),
        ("whisper-distill", "openai", "whisper-distill", 0.82),
        ("nvidia-parakeet", "nvidia", "parakeet", 0.86),
        ("nvidia-canary", "nvidia", "canary", 0.85),
        ("seamlessm4t", "meta", "seamlessm4t", 0.8),
        ("funasr", "alibaba", "funasr", 0.82),
        ("espnet-asr", "espnet", "espnet", 0.78),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Audio,
            vec![Task::Speech],
            vec![SpecialistLane::Speech],
            CapabilityProfile {
                speech_to_text: score,
                local_deployability: 0.85,
                cost_efficiency: 0.9,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Speech, score)],
        ));
    }

    for (id, provider, model, score) in [
        ("elevenlabs-tts", "elevenlabs", "tts", 0.97),
        ("openai-audio-tts", "openai", "tts", 0.94),
        ("google-tts", "google", "tts", 0.9),
        ("cartesia", "cartesia", "sonic", 0.94),
        ("playht", "playht", "playht", 0.88),
        ("azure-neural-voice", "azure", "neural-voice", 0.9),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Audio,
            vec![Task::Voice],
            vec![SpecialistLane::Voice],
            CapabilityProfile {
                text_to_speech: score,
                latency: 0.7,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Voice, score)],
        ));
    }

    for (id, provider, model, score) in [
        ("kokoro", "hexgrad", "kokoro", 0.84),
        ("chatterbox", "resemble", "chatterbox", 0.84),
        ("xtts", "coqui", "xtts", 0.82),
        ("fish-speech", "fish-audio", "fish-speech", 0.86),
        ("styletts", "community", "styletts", 0.78),
        ("bark", "suno", "bark", 0.76),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Audio,
            vec![Task::Voice],
            vec![SpecialistLane::Voice],
            CapabilityProfile {
                text_to_speech: score,
                local_deployability: 0.85,
                cost_efficiency: 0.9,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Voice, score)],
        ));
    }

    for (id, provider, model, score, closed) in [
        ("suno", "suno", "suno", 0.95, true),
        ("udio", "udio", "udio", 0.95, true),
        ("stable-audio-api", "stability", "stable-audio-api", 0.88, true),
        ("elevenlabs-sfx", "elevenlabs", "sfx", 0.87, true),
        ("musicgen", "meta", "musicgen", 0.78, false),
        ("audiocraft", "meta", "audiocraft", 0.78, false),
        ("stable-audio-open", "stability", "stable-audio-open", 0.8, false),
        ("ace-step", "ace", "ace-step", 0.82, false),
        ("tangoflux", "declare-lab", "tangoflux", 0.78, false),
    ] {
        let tasks = vec![Task::Music, Task::Creative];
        let lanes = vec![SpecialistLane::Music];
        let caps = CapabilityProfile {
            music_generation: score,
            cost_efficiency: if closed { 0.4 } else { 0.85 },
            local_deployability: if closed { 0.0 } else { 0.8 },
            ..CapabilityProfile::default()
        };
        let card = if closed {
            closed_cloud(id, provider, model, Modality::Music, tasks, lanes, caps, &[(SpecialistLane::Music, score)])
        } else {
            open_weight(id, provider, model, Modality::Music, tasks, lanes, caps, &[(SpecialistLane::Music, score)])
        };
        models.push(card);
    }
}

fn add_safety_and_judges(models: &mut Vec<ModelCard>) {
    for (id, provider, model, score) in [
        ("openai-moderation", "openai", "moderation", 0.95),
        ("google-safety", "google", "safety", 0.92),
        ("anthropic-safety", "anthropic", "safety", 0.92),
        ("perspective-api", "google", "perspective-api", 0.86),
    ] {
        models.push(closed_cloud(
            id,
            provider,
            model,
            Modality::Safety,
            vec![Task::Safety],
            vec![SpecialistLane::Safety],
            CapabilityProfile {
                safety_filter: score,
                latency: 0.8,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Safety, score)],
        ));
    }

    for (id, provider, model, score) in [
        ("llama-guard", "meta", "llama-guard", 0.86),
        ("shieldgemma", "google", "shieldgemma", 0.85),
        ("wildguard", "allenai", "wildguard", 0.83),
        ("detoxify", "unitary", "detoxify", 0.75),
        ("presidio", "microsoft", "presidio", 0.86),
        ("prompt-injection-detector", "community", "prompt-injection-detector", 0.78),
    ] {
        models.push(open_weight(
            id,
            provider,
            model,
            Modality::Safety,
            vec![Task::Safety],
            vec![SpecialistLane::Safety],
            CapabilityProfile {
                safety_filter: score,
                local_deployability: 0.9,
                cost_efficiency: 0.9,
                ..CapabilityProfile::default()
            },
            &[(SpecialistLane::Safety, score)],
        ));
    }

    for (id, provider, model, score, closed) in [
        ("gpt-5.5-judge", "openai", "gpt-5.5", 0.98, true),
        ("claude-opus-4.8-judge", "anthropic", "claude-opus-4.8", 0.97, true),
        ("gemini-pro-judge", "google", "gemini-3.1-pro", 0.95, true),
        ("glm-5.2-judge", "zai", "glm-5.2", 0.91, false),
        ("qwen-reasoning-judge", "qwen", "qwen3.5-397b-reasoning", 0.9, false),
        ("kimi-thinking-judge", "moonshot", "kimi-k2.7-thinking", 0.91, false),
        ("deepseek-reasoning-judge", "deepseek", "deepseek-v4-pro-max", 0.92, false),
        ("prometheus-judge", "prometheus", "prometheus", 0.84, false),
        ("judgelm", "community", "judgelm", 0.8, false),
    ] {
        let tasks = vec![Task::Judge, Task::Reasoning];
        let lanes = vec![SpecialistLane::Judge, SpecialistLane::Reasoning];
        let caps = CapabilityProfile {
            judge_quality: score,
            reasoning_depth: score,
            text_generation: 0.6,
            ..CapabilityProfile::default()
        };
        let card = if closed {
            closed_cloud(id, provider, model, Modality::Judge, tasks, lanes, caps, &[(SpecialistLane::Judge, score)])
        } else {
            open_weight(id, provider, model, Modality::Judge, tasks, lanes, caps, &[(SpecialistLane::Judge, score)])
        };
        models.push(card);
    }
}
