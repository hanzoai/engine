#[derive(Clone, Debug)]
pub enum SpeculativeAttachKind {
    Mtp {
        assistant: String,
        n_predict: usize,
    },
    DraftModel {
        gamma: usize,
    },
    Dspark {
        block_size: usize,
        confidence_threshold: f32,
    },
    PromptLookup {
        ngram_min: usize,
        ngram_max: usize,
        gamma: usize,
    },
}

#[derive(Clone, Debug)]
pub struct SpeculativeAttachInfo {
    pub kind: SpeculativeAttachKind,
}

impl SpeculativeAttachInfo {
    pub fn mtp(assistant: String, n_predict: usize) -> Self {
        Self {
            kind: SpeculativeAttachKind::Mtp {
                assistant,
                n_predict,
            },
        }
    }

    pub fn draft_model(gamma: usize) -> Self {
        Self {
            kind: SpeculativeAttachKind::DraftModel { gamma },
        }
    }

    pub fn dspark(block_size: usize, confidence_threshold: f32) -> Self {
        Self {
            kind: SpeculativeAttachKind::Dspark {
                block_size,
                confidence_threshold,
            },
        }
    }

    pub fn prompt_lookup(ngram_min: usize, ngram_max: usize, gamma: usize) -> Self {
        Self {
            kind: SpeculativeAttachKind::PromptLookup {
                ngram_min,
                ngram_max,
                gamma,
            },
        }
    }
}

pub fn log_attach(info: &SpeculativeAttachInfo) {
    match &info.kind {
        SpeculativeAttachKind::Mtp {
            assistant,
            n_predict,
        } => tracing::info!(
            "Speculative decoding enabled: MTP assistant `{assistant}` with n_predict={n_predict}"
        ),
        SpeculativeAttachKind::DraftModel { gamma } => tracing::info!(
            "Speculative decoding enabled: classic draft+target with gamma={gamma}"
        ),
        SpeculativeAttachKind::Dspark {
            block_size,
            confidence_threshold,
        } => tracing::info!(
            "Speculative decoding enabled: DSpark parallel-block draft with block_size={block_size}, confidence_threshold={confidence_threshold}"
        ),
        SpeculativeAttachKind::PromptLookup {
            ngram_min,
            ngram_max,
            gamma,
        } => tracing::info!(
            "Speculative decoding enabled: prompt-lookup n-gram draft with ngram={ngram_min}..={ngram_max}, gamma={gamma}"
        ),
    }
}
