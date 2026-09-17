use std::sync::Arc;

use hanzo_ml::{Result, Tensor};

use super::{
    logging::log_attach, SpeculativeAttachInfo, SpeculativeConfig, SpeculativeProposalBatch,
    SpeculativeProposeBatchCtx,
};

/// A target's token embedding (`ids -> [.., hidden]`) and output head
/// (`[.., hidden] -> [.., vocab]`), lent to a draft that carries neither. Decoding the
/// draft through the target's own head puts its logits on the verifier's scale.
#[derive(Clone)]
pub struct SpeculativeSharedHeads {
    pub embed: SharedLayer,
    pub lm_head: SharedLayer,
}

/// One target layer, lent as a function of its input.
pub type SharedLayer = Arc<dyn Fn(&Tensor) -> Result<Tensor> + Send + Sync>;

pub trait SpeculativeTargetMixin {
    fn attach_speculative(
        &mut self,
        config: SpeculativeConfig,
    ) -> Result<Option<SpeculativeAttachInfo>> {
        match config {
            SpeculativeConfig::Off => Ok(None),
            _ => hanzo_ml::bail!("This model does not support speculative decoding."),
        }
    }

    fn log_speculative_attach(&self, info: &SpeculativeAttachInfo) {
        log_attach(info);
    }

    fn has_speculative_proposer(&self) -> bool {
        false
    }

    fn speculative_proposal_len(&self) -> Option<usize> {
        None
    }

    /// Returns `Ok(None)` when speculation is unsupported for the current step.
    /// Return `Err` only for real failures that should stop generation.
    fn speculative_propose(
        &mut self,
        _ctx: SpeculativeProposeBatchCtx<'_>,
    ) -> Result<Option<SpeculativeProposalBatch>> {
        Ok(None)
    }

    /// Returns `Ok(None)` when the active proposer does not need target hidden state.
    /// Return `Err` only when hidden state was expected but unavailable or invalid.
    fn speculative_target_hiddens(&self, _rows: &[(usize, usize)]) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// Names the sequences the next forward runs, so the captured hidden prefix stays
    /// attributed to one sequence. Default no-op.
    fn note_speculative_forward(&self, _seq_ids: &[usize]) {}

    /// The embedding and output head a headless draft (DFlash) decodes through.
    /// `None` when the model does not lend them (the default).
    fn speculative_shared_heads(&self) -> Option<SpeculativeSharedHeads> {
        None
    }

    /// Capture the output of every layer in `layers` (a draft checkpoint's `target_layer_ids`)
    /// during each forward, keeping the last `retain` positions, or all of them for `None`.
    /// Default no-op: only models that expose multi-layer hiddens override it. Uses interior
    /// mutability, so `&self` suffices.
    fn set_speculative_capture_layers(&self, _layers: Vec<usize>, _retain: Option<usize>) {}

    /// The multi-layer target hiddens captured so far for the one running sequence.
    /// `Ok(None)` when capture is off, unsupported (the default), or holds nothing.
    fn speculative_target_hidden_layers(
        &self,
        _rows: &[(usize, usize)],
    ) -> Result<Option<super::HiddenWindow>> {
        Ok(None)
    }
}
