use std::sync::{Arc, Mutex};

use hanzo_ml::{Result, Tensor};
use rand_isaac::Isaac64Rng;

use crate::pipeline::text_models_inputs_processor::PagedAttentionMeta;
use crate::sequence::Sequence;

pub type TargetTokenEmbedder<'a> = dyn Fn(&Tensor) -> Result<Tensor> + 'a;

pub enum SpeculativeKvCache<'a> {
    Paged {
        metadata: &'a PagedAttentionMeta,
        kv_cache: &'a [(Tensor, Tensor)],
    },
    /// Normal (non-paged) KV cache. Self-speculative proposers that keep their own
    /// draft KV (e.g. DeepSeek-V4 MTP) don't read the target tensors — the confirmed
    /// tokens arrive via `sampled_tokens`/`sequences` — so this variant carries none.
    Normal,
}

pub struct SpeculativeProposeBatchCtx<'a> {
    pub sampled_tokens: &'a [u32],
    pub sampled_tokens_emitted: bool,
    pub seq_ids: &'a [usize],
    pub base_lens: &'a [usize],
    pub sequences: &'a [&'a Sequence],
    pub cache: SpeculativeKvCache<'a>,
    pub target_hiddens: Option<Tensor>,
    /// Multi-layer target hidden prefix for DSpark-style parallel-block proposers: one
    /// tensor per fused target decoder layer, each `[prefix_len, hidden]`. `None` for
    /// single-hidden proposers (V4 MTP) and standalone draft models — additive, so
    /// existing proposers are unaffected.
    pub target_hidden_layers: Option<Vec<Tensor>>,
    pub rng: Arc<Mutex<Isaac64Rng>>,
}

#[derive(Clone, Debug)]
pub struct SpeculativeProposal {
    pub tokens: Vec<u32>,
    pub logits: Option<Tensor>,
}

impl SpeculativeProposal {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self {
            tokens,
            logits: None,
        }
    }

    pub fn with_logits(tokens: Vec<u32>, logits: Tensor) -> Self {
        Self {
            tokens,
            logits: Some(logits),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

pub struct SpeculativeProposalBatch {
    pub proposals: Vec<SpeculativeProposal>,
}

impl SpeculativeProposalBatch {
    pub fn new(proposals: Vec<SpeculativeProposal>) -> Self {
        Self { proposals }
    }
}

pub trait SpeculativeProposer {
    fn proposal_len(&self) -> usize;

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch>;

    /// Drop per-sequence proposer state for sequences no longer live. `live` is the set of
    /// still-running sequence ids; anything keyed off a finished sequence must be released so a
    /// long-running server doesn't leak. Default no-op: proposers that hold no per-seq state
    /// (e.g. MTP) need not implement it.
    fn retain_seqs(&mut self, _live: &[usize]) {}
}
