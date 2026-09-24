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
    Dflash {
        block_size: usize,
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
    /// The drafter's name as requests and the request ledger spell it: `mtp`, `dflash`, or
    /// `spec` for a separate draft model, DSpark and prompt lookup (Halogen spec §10.1, §11).
    pub fn name(&self) -> &'static str {
        match self.kind {
            SpeculativeAttachKind::Mtp { .. } => "mtp",
            SpeculativeAttachKind::Dflash { .. } => "dflash",
            SpeculativeAttachKind::DraftModel { .. }
            | SpeculativeAttachKind::Dspark { .. }
            | SpeculativeAttachKind::PromptLookup { .. } => "spec",
        }
    }

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

    pub fn dflash(block_size: usize) -> Self {
        Self {
            kind: SpeculativeAttachKind::Dflash { block_size },
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
        SpeculativeAttachKind::Dflash { block_size } => tracing::info!(
            "Speculative decoding enabled: DFlash 2 block-diffusion draft with block_size={block_size}"
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

#[cfg(test)]
mod tests {
    use super::SpeculativeAttachInfo;

    #[test]
    fn drafter_names() {
        assert_eq!(SpeculativeAttachInfo::mtp("self".into(), 3).name(), "mtp");
        assert_eq!(SpeculativeAttachInfo::dflash(16).name(), "dflash");
        assert_eq!(SpeculativeAttachInfo::draft_model(4).name(), "spec");
        assert_eq!(SpeculativeAttachInfo::dspark(16, 0.0).name(), "spec");
        assert_eq!(SpeculativeAttachInfo::prompt_lookup(3, 7, 7).name(), "spec");
    }
}
