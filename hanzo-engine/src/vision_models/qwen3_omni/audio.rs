#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, dead_code)]

use std::sync::Arc;

use hanzo_ml::{Result, Tensor};
use hanzo_quant::{QuantMethod, ShardedVarBuilder};

use crate::layers;
use crate::vision_models::voxtral::config::WhisperEncoderArgs;

/// STUB audio encoder for the Omni scaffold.
///
/// The real Omni audio path reuses the Voxtral/Whisper causal encoder
/// (`vision_models::voxtral::encoder::VoxtralEncoder`): two Conv1d mel projections, RoPE,
/// sliding-window causal attention, SwiGLU, RMSNorm, followed by a temporal adapter. That encoder
/// is currently a private module inside `voxtral`, and Omni has no published weights, so this stub
/// stands in: it exposes the same forward contract (mel features -> hidden states at the Thinker
/// width) via a single learned projection, so the audio -> Thinker bridge has a concrete shape to
/// route through. Replace with the real encoder + temporal adapter when wiring real weights.
pub struct OmniAudioEncoderStub {
    proj: Arc<dyn QuantMethod>,
    out_dim: usize,
}

impl OmniAudioEncoderStub {
    pub fn new(
        cfg: &WhisperEncoderArgs,
        thinker_hidden_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let proj = layers::linear_no_bias(
            cfg.audio_encoding_args.num_mel_bins,
            thinker_hidden_size,
            vb.pp("audio_proj"),
        )?;
        let proj: Arc<dyn QuantMethod> =
            Arc::new(hanzo_quant::UnquantLinear::new(
                hanzo_quant::QuantMethodConfig::Unquantized(proj),
            )?);
        Ok(Self {
            proj,
            out_dim: thinker_hidden_size,
        })
    }

    /// mel features (b, t, n_mels) -> audio embeds (b, t, thinker_hidden). PLACEHOLDER feed.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        self.proj.forward(mel)
    }

    pub fn out_dim(&self) -> usize {
        self.out_dim
    }
}
