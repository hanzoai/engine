//! Request → [`Task`]. Heuristic today; the `Classifier` seam lets a learned
//! model drop in without touching the policy (same as the Python router).

use crate::registry::{Modality, Task};

/// The features the router reasons about — extracted from the request by the
/// caller (hanzo-node), kept as a value so classification is pure.
#[derive(Debug, Clone, Default)]
pub struct Request {
    /// Concatenated user text (last user turn is enough for routing).
    pub text: String,
    /// Approximate prompt length in tokens (drives long-context routing).
    pub approx_tokens: usize,
    /// Request carries image/video input.
    pub has_media: bool,
    /// Optional explicit task override (skips classification).
    pub task_hint: Option<Task>,
    /// Desired output modality (image-gen/dub set this; chat leaves it None).
    pub modality_hint: Option<Modality>,
}

impl Request {
    /// The pool this request must be served from: an explicit hint wins, else
    /// media input implies vision, else text.
    pub fn target_modality(&self) -> Modality {
        self.modality_hint.unwrap_or({
            if self.has_media {
                Modality::Vision
            } else {
                Modality::Text
            }
        })
    }
}

pub trait Classifier {
    fn classify(&self, req: &Request) -> Task;
}

/// Keyword/length/media heuristics. Mirrors the Python `HeuristicClassifier`.
#[derive(Debug, Clone, Default)]
pub struct Heuristic;

impl Classifier for Heuristic {
    fn classify(&self, req: &Request) -> Task {
        if let Some(t) = req.task_hint {
            return t;
        }
        if req.has_media {
            return Task::Vision;
        }
        if req.approx_tokens >= 32_000 {
            return Task::LongContext;
        }
        let t = req.text.to_lowercase();
        let has = |words: &[&str]| words.iter().any(|w| t.contains(w));
        if t.contains("```")
            || has(&[
                "code",
                "function",
                "bug",
                "compile",
                "stack trace",
                "refactor",
            ])
        {
            Task::Code
        } else if has(&[
            "prove",
            "theorem",
            "integral",
            "equation",
            "calculate",
            "solve for",
        ]) {
            Task::Math
        } else if has(&[
            "why",
            "reason",
            "step by step",
            "analyze",
            "explain how",
            "trade-off",
        ]) {
            Task::Reasoning
        } else if has(&["write a story", "poem", "creative", "imagine", "brainstorm"]) {
            Task::Creative
        } else if req.approx_tokens <= 16 {
            Task::CheapChat
        } else {
            Task::General
        }
    }
}
