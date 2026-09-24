//! Server-wide request defaults: a value a request omits comes from here, and a value the request
//! sends always wins (Halogen spec §1.2, §5).

/// Defaults applied to generation requests, injected as an axum `Extension`.
#[derive(Clone, Debug)]
pub struct Defaults {
    /// The model id `/v1/models` lists and every response carries; `None` keeps the engine's id.
    pub served_name: Option<String>,
    /// Token budget (reasoning plus answer) for chat, Responses and Messages requests that send none.
    pub max_tokens: usize,
    /// Token budget for `/v1/completions` requests that send none.
    pub completion_max_tokens: usize,
    /// Largest budget a request may ask for; more is refused, never clamped.
    pub max_tokens_cap: usize,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<usize>,
    pub min_p: Option<f64>,
    pub presence_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    /// One of minimal, low, medium, high, xhigh.
    pub reasoning_effort: Option<String>,
    pub enable_thinking: Option<bool>,
    /// Think-block budget for requests that send none.
    pub max_thinking_tokens: Option<usize>,
    /// Tokens kept for the answer when the think-block budget is derived from `max_tokens`.
    /// `None` is the policy `max(1024, 15% of max_tokens)`; `Some(0)` keeps no room.
    pub answer_room: Option<usize>,
}

impl Default for Defaults {
    fn default() -> Self {
        Self {
            served_name: None,
            max_tokens: 8192,
            completion_max_tokens: 128,
            max_tokens_cap: 65536,
            temperature: None,
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            reasoning_effort: None,
            enable_thinking: None,
            max_thinking_tokens: None,
            answer_room: None,
        }
    }
}
