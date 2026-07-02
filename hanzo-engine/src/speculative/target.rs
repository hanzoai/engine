use hanzo_ml::{Result, Tensor};

use super::{
    logging::log_attach, SpeculativeAttachInfo, SpeculativeConfig, SpeculativeProposalBatch,
    SpeculativeProposeBatchCtx,
};

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

    /// Enable capture of the DSpark target-layer hidden states, stashing every layer index in
    /// `layers` (the draft checkpoint's `target_layer_ids`) during each forward. Default no-op:
    /// only models that expose multi-layer hiddens (Qwen3) override it. Uses interior
    /// mutability, so `&self` suffices.
    fn set_speculative_capture_layers(&self, _layers: Vec<usize>) {}

    /// The multi-layer target hidden prefix captured by the most recent forward, one
    /// `[prefix_len, hidden]` tensor per fused layer, gathered for the requested `(seq, row)`
    /// pairs. `Ok(None)` when capture is off or unsupported (the default).
    fn speculative_target_hidden_layers(
        &self,
        _rows: &[(usize, usize)],
    ) -> Result<Option<Vec<Tensor>>> {
        Ok(None)
    }
}
