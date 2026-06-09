#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, dead_code)]

//! Qwen3.6-MoE-Omni: the fused understand -> translate -> speak model (SCAFFOLD).
//!
//! This module ASSEMBLES existing hanzo building blocks into one omni forward. It is a compiling
//! structural scaffold, NOT a working model: no `zenlm/zen-omni` weights are published, the
//! audio-encoder -> Thinker bridge and the Thinker -> Talker hidden-state bridge are placeholders,
//! and nothing here has been trained or numerically validated. See module-level notes and the
//! `BRIDGE PLACEHOLDER` markers below.
//!
//! Assembled real pieces:
//! - Thinker: [`crate::vision_models::qwen3_5_moe::Qwen3_5MoeModel`] (text + Qwen3-VL vision tower +
//!   hybrid full/linear MoE decoder). This is the same backbone used for Qwen3.5/3.6-VL-MoE.
//! - Vision encoder: [`crate::vision_models::qwen3_vl::vision::Qwen3VLVisionModel`], reached through
//!   the Thinker (the Thinker already owns and routes it for image/video).
//! - Audio encoder: a STUB ([`audio::OmniAudioEncoderStub`]); the real path reuses the
//!   Voxtral/Whisper causal encoder.
//! - Talker: [`crate::speech_models::qwen3_tts::Talker`] + [`crate::speech_models::qwen3_tts::CodePredictor`]
//!   (the TTS codec backbone) for the speech output head.

use std::any::Any;
use std::sync::Arc;

use hanzo_ml::{Device, Result, Tensor};
use hanzo_quant::{QuantMethod, ShardedVarBuilder};

use crate::layers;
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{
    EitherCache, IsqModel, MultimodalModel, NormalLoadingMetadata, SupportedModality,
};
use crate::vision_models::qwen3_5_moe::Qwen3_5MoeModel;
use crate::vision_models::qwen3_vl::Qwen3VLVisionSpecificArgs;

use crate::speech_models::qwen3_tts::{CodePredictor, Talker};

pub(crate) mod audio;
pub(crate) mod config;

pub(crate) use config::Qwen3OmniConfig;
use audio::OmniAudioEncoderStub;

/// The four input modalities and two output modalities Omni declares to the engine.
pub fn omni_modalities() -> crate::pipeline::Modalities {
    crate::pipeline::Modalities {
        input: vec![
            SupportedModality::Text,
            SupportedModality::Vision,
            SupportedModality::Audio,
            SupportedModality::Video,
        ],
        output: vec![SupportedModality::Text, SupportedModality::Audio],
    }
}

/// Speech head: the Talker backbone, its MTP code predictor, and the projection that bridges
/// Thinker hidden states into the Talker input space.
pub struct OmniSpeechHead {
    talker: Talker,
    code_predictor: CodePredictor,
    /// BRIDGE PLACEHOLDER: Thinker hidden (d_thinker) -> Talker hidden (d_talker).
    /// In a real Omni this is a trained adapter (often per-layer / with a small projector stack and
    /// the talker's own text+codec embedding interleave). Here it is a single untrained linear.
    thinker_to_talker: Arc<dyn QuantMethod>,
}

impl OmniSpeechHead {
    fn new(
        cfg: &Qwen3OmniConfig,
        thinker_hidden_size: usize,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let talker = Talker::new(&cfg.talker_config.talker_config, vb.pp("talker"))?;
        let code_predictor = CodePredictor::new(
            &cfg.talker_config.talker_config.code_predictor_config,
            cfg.talker_config.talker_config.hidden_size,
            vb.pp("talker").pp("code_predictor"),
        )?;
        let proj = layers::linear_no_bias(
            thinker_hidden_size,
            cfg.talker_config.talker_config.hidden_size,
            vb.pp("thinker_to_talker"),
        )?;
        let thinker_to_talker: Arc<dyn QuantMethod> = Arc::new(
            hanzo_quant::UnquantLinear::new(hanzo_quant::QuantMethodConfig::Unquantized(proj))?,
        );
        Ok(Self {
            talker,
            code_predictor,
            thinker_to_talker,
        })
    }

    /// BRIDGE PLACEHOLDER. Maps Thinker hidden states into the Talker embedding space.
    /// A real implementation interleaves talker text/codec embeddings and runs the AR talker loop;
    /// here we only project so the route Thinker -> Talker is type-correct and exercised.
    pub fn project_thinker_hidden(&self, thinker_hidden: &Tensor) -> Result<Tensor> {
        self.thinker_to_talker.forward(thinker_hidden)
    }

    pub fn talker(&self) -> &Talker {
        &self.talker
    }

    pub fn code_predictor(&self) -> &CodePredictor {
        &self.code_predictor
    }
}

/// Extra inputs the Omni forward accepts beyond the Thinker's vision args: optional audio mel
/// features and their placeholder spans in the token sequence.
pub struct Qwen3OmniSpecificArgs {
    /// The Thinker's existing vision/text routing args (image/video grids, pad spans, etc.).
    pub thinker_args: Qwen3VLVisionSpecificArgs,
    /// Mel spectrogram features (b, t, n_mels) when audio is present.
    pub audio_features: Option<Tensor>,
    /// (batch, [(start, end)]) spans of audio placeholder tokens to scatter audio embeds into.
    pub continuous_audio_pad: Vec<Vec<(usize, usize)>>,
}

pub struct Qwen3OmniModel {
    thinker: Qwen3_5MoeModel,
    audio_encoder: OmniAudioEncoderStub,
    speech_head: OmniSpeechHead,
    audio_token_id: u32,
    device: Device,
}

impl Qwen3OmniModel {
    pub fn new(
        cfg: &Qwen3OmniConfig,
        vb: ShardedVarBuilder,
        is_gptx: bool,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        let device = normal_loading_metadata.real_device.clone();
        let thinker_hidden_size = cfg.thinker_config.text_config.hidden_size;

        let thinker = Qwen3_5MoeModel::new(
            &cfg.thinker_config,
            vb.pp("thinker"),
            is_gptx,
            normal_loading_metadata,
            attention_mechanism,
        )?;

        let audio_encoder = OmniAudioEncoderStub::new(
            &cfg.audio_encoder_config,
            thinker_hidden_size,
            vb.pp("audio_encoder"),
        )?;

        let speech_head = OmniSpeechHead::new(cfg, thinker_hidden_size, vb.clone())?;

        Ok(Self {
            thinker,
            audio_encoder,
            speech_head,
            audio_token_id: cfg.audio_token_id,
            device,
        })
    }

    /// Encode audio mel features into Thinker-width embeds. BRIDGE PLACEHOLDER: the stub encoder is
    /// not a trained Whisper stack, and the scatter into the token sequence is not yet performed
    /// (the Thinker's `forward_embeds` does not currently accept an extra audio-embed scatter list,
    /// so audio is encoded and dropped here pending the real bridge).
    fn encode_audio(&self, audio_features: &Tensor) -> Result<Tensor> {
        self.audio_encoder.forward(audio_features)
    }

    /// Omni forward skeleton: route the multimodal inputs through the Thinker, optionally encode
    /// audio, and return the Thinker's text logits. The speech (Talker) branch is reachable via
    /// [`Self::generate_speech_from_hidden`] but is NOT driven from here yet.
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        audio_features: Option<Tensor>,
        thinker_args: Qwen3VLVisionSpecificArgs,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        if let Some(audio) = audio_features.as_ref() {
            // BRIDGE PLACEHOLDER: audio is encoded to Thinker width then dropped (see encode_audio).
            let _audio_embeds = self.encode_audio(audio)?;
        }

        // Vision + video + text all route through the Thinker (it owns the Qwen3-VL vision tower).
        MultimodalModel::forward(
            &self.thinker,
            input_ids,
            pixel_values,
            seqlen_offsets,
            context_lens,
            position_ids,
            Box::new(thinker_args),
            metadata,
            flash_params,
        )
    }

    /// SPEECH OUTPUT (scaffold): given Thinker hidden states for the response, project them into the
    /// Talker space. The full path (AR talker loop -> MTP code predictor -> neural codec -> PCM)
    /// lives in `speech_models::qwen3_tts` and is not invoked here. THINKER -> TALKER BRIDGE
    /// PLACEHOLDER: returns the projected hidden states only.
    pub fn generate_speech_from_hidden(&self, thinker_hidden: &Tensor) -> Result<Tensor> {
        self.speech_head.project_thinker_hidden(thinker_hidden)
    }

    pub fn modalities(&self) -> crate::pipeline::Modalities {
        omni_modalities()
    }

    pub fn speech_head(&self) -> &OmniSpeechHead {
        &self.speech_head
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl crate::speculative::SpeculativeTargetMixin for Qwen3OmniModel {}

impl MultimodalModel for Qwen3OmniModel {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> Result<Tensor> {
        let Qwen3OmniSpecificArgs {
            thinker_args,
            audio_features,
            continuous_audio_pad: _,
        } = *model_specific_args
            .downcast()
            .expect("Cannot downcast into `Qwen3OmniSpecificArgs`");
        self.forward(
            input_ids,
            pixel_values,
            audio_features,
            thinker_args,
            seqlen_offsets,
            context_lens,
            position_ids,
            metadata,
            flash_params,
        )
    }
    fn cache(&self) -> &EitherCache {
        self.thinker.cache()
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        self.thinker.cache_mut()
    }
    fn device(&self) -> &Device {
        MultimodalModel::device(&self.thinker)
    }
    fn max_seq_len(&self) -> usize {
        self.thinker.max_seq_len()
    }
    fn config(&self) -> &ModelConfigMetadata {
        self.thinker.config()
    }
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any> {
        let thinker_args = *self
            .thinker
            .default_model_specific_args(input_ids)
            .downcast::<Qwen3VLVisionSpecificArgs>()
            .expect("Thinker default args are Qwen3VLVisionSpecificArgs");
        Box::new(Qwen3OmniSpecificArgs {
            thinker_args,
            audio_features: None,
            continuous_audio_pad: vec![],
        })
    }
}

impl IsqModel for Qwen3OmniModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn crate::device_map::DeviceMapper,
    ) {
        self.thinker.get_layers()
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        self.thinker.residual_tensors()
    }
}

impl crate::amoe::AnyMoeBaseModelMixin for Qwen3OmniModel {}
