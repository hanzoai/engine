#![allow(clippy::cast_possible_truncation)]

//! Input processor for Qwen3-Omni serving.
//!
//! Turns a chat request into `(input_ids, OmniSpecificArgs)` for [`super::Qwen3OmniModel`]'s
//! [`crate::pipeline::MultimodalModel`] forward. **Audio** is wired end-to-end (the Omni
//! differentiator): each `<|AUDIO|>` placeholder the chat template emits is expanded to the right
//! number of Thinker tokens ([`super::omni_audio_feat_len`]) and the raw waveform is turned into the
//! Whisper-style log-mel the validated audio tower consumes (reusing
//! [`Qwen3AsrAudioProcessor`]) and handed through as an [`super::OmniSpecificArgs`] payload.
//!
//! The two-phase contract mirrors the established audio processor ([`crate::vision_models::voxtral`]):
//! the first (scheduling) pass rewrites the prompt tokens — expanding placeholders so the KV cache is
//! sized correctly — and the second (prefill) pass materializes the mel payloads. Text-only requests
//! fall straight through to the standard text input path, preserving validated text serving.
//!
//! Vision (image/video) input is **not** expanded here yet: the model side is ready (the vision tower
//! is validated and registered as a [`super::modality::ModalityEncoder`], and 3D mRoPE serving is
//! wired via [`super::omni_get_rope_index`] + `forward_cached_mrope`), but the processor still needs
//! the shared Qwen3-VL image preprocessor exposed and `VisionModality` extended to accept an explicit
//! `grid_thw` for non-square images. Until then this processor advertises Text + Audio.

use std::{any::Any, sync::Arc};

use anyhow::Result;
use hanzo_ml::Device;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    speech_models::qwen3_asr::Qwen3AsrAudioProcessor,
    vision_models::ModelInputs,
};

use super::config::OmniAudioConfig;
use super::modality::ModalityInput;
use super::{omni_audio_feat_len, OmniSpecificArgs};

/// Whisper-standard log-mel frontend parameters for the Omni audio tower (n_fft=400, hop=160,
/// 16 kHz, 128 mel bins) — the same values the Qwen3-ASR audio config defaults to.
fn build_audio_processor(cfg: &OmniAudioConfig) -> Qwen3AsrAudioProcessor {
    use crate::speech_models::qwen3_asr::config::AudioEncoderConfig;
    let enc = AudioEncoderConfig {
        d_model: cfg.d_model,
        num_layers: cfg.encoder_layers,
        num_heads: cfg.encoder_attention_heads,
        ffn_dim: cfg.encoder_ffn_dim,
        n_mels: cfg.num_mel_bins,
        conv_channels: cfg.downsample_hidden_size,
        output_dim: cfg.output_dim,
        sampling_rate: 16_000,
        hop_length: 160,
        window_size: 400,
        n_window: cfg.n_window,
        n_window_infer: cfg.n_window_infer,
    };
    Qwen3AsrAudioProcessor::new(&enc)
}

/// [`Processor`] for Qwen3-Omni: the default `process` applies the Qwen chat template + tokenizes
/// (modality tokens come from the template), and [`Self::inputs_processor`] builds the modality
/// payloads.
pub struct Qwen3OmniProcessor {
    audio_token_id: u32,
    audio_config: OmniAudioConfig,
}

impl Qwen3OmniProcessor {
    pub fn new(audio_token_id: u32, audio_config: OmniAudioConfig) -> Self {
        Self {
            audio_token_id,
            audio_config,
        }
    }
}

impl Processor for Qwen3OmniProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Qwen3OmniInputsProcessor {
            audio_token_id: self.audio_token_id,
            audio: build_audio_processor(&self.audio_config),
        })
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        // Keep the structured message content so the chat template can place modality tokens.
        MessagesAction::Keep
    }
}

struct Qwen3OmniInputsProcessor {
    audio_token_id: u32,
    audio: Qwen3AsrAudioProcessor,
}

impl InputsProcessor for Qwen3OmniInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }

    #[allow(clippy::too_many_arguments)]
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        return_raw_logits: bool,
        sliding_window: Option<usize>,
        _other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        if is_xlora {
            return Err(anyhow::Error::msg("Cannot make inputs for X-LoRA vision model."));
        }
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        if tokenizer.is_none() {
            return Err(anyhow::Error::msg(
                "Qwen3OmniInputsProcessor requires a specified tokenizer.",
            ));
        }

        // ── Audio ──────────────────────────────────────────────────────────────────────────────
        // Phase 1 (scheduling): expand each `<|AUDIO|>` placeholder to its Thinker-token length so
        // the KV cache is sized correctly. Phase 2 (prefill): materialize the mel payloads.
        let mut payloads: Vec<(u32, ModalityInput)> = Vec::new();
        let mut audio_seqlens: Vec<usize> = Vec::new();
        if is_prompt {
            for seq in input_seqs.iter_mut() {
                if !seq.multimodal.has_changed_prompt {
                    // Phase 1: expand each `<|AUDIO|>` to its Thinker-token length, then resize the
                    // KV allocation. (Mirrors the Voxtral processor's prompt-rewrite phase.)
                    if seq.has_audios() {
                        let audios = seq.multimodal.clone_audios().unwrap_or_default();
                        let toks = seq.get_toks().to_vec();
                        let mut new_toks = Vec::with_capacity(toks.len());
                        let mut ai = 0usize;
                        for &t in &toks {
                            if t == self.audio_token_id {
                                let audio = audios.get(ai).ok_or_else(|| {
                                    anyhow::Error::msg(
                                        "more <|AUDIO|> placeholders than provided audios",
                                    )
                                })?;
                                let mel = self
                                    .audio
                                    .process(audio, device)
                                    .map_err(anyhow::Error::msg)?;
                                let len = omni_audio_feat_len(mel.dims()[2]);
                                new_toks.extend(std::iter::repeat_n(self.audio_token_id, len));
                                ai += 1;
                            } else {
                                new_toks.push(t);
                            }
                        }
                        seq.set_toks_and_reallocate(new_toks, paged_attn_metadata.as_mut());
                        seq.multimodal.has_changed_prompt = true;
                    }
                } else if let Some(audios) = seq.take_audios() {
                    // Phase 2 (prefill): materialize the mel payloads consumed by `fuse_modalities`.
                    for audio in &audios {
                        let mel = self.audio.process(audio, device).map_err(anyhow::Error::msg)?;
                        audio_seqlens.push(mel.dims()[2]);
                        payloads.push((self.audio_token_id, ModalityInput::Audio(mel)));
                    }
                }
            }
        }

        // ── Standard text input assembly (shared with every text/vision model) ──────────────────
        let text_models_inputs_processor::InnerInputProcessorOutput {
            inputs:
                text_models_inputs_processor::InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = if is_prompt {
            get_prompt_input(
                input_seqs.iter().map(|seq| seq.get_toks()).collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )
            .map_err(anyhow::Error::msg)?
        } else {
            get_completion_input(
                input_seqs.iter().map(|seq| seq.get_toks()).collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                mapper,
                sliding_window,
            )
            .map_err(anyhow::Error::msg)?
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(OmniSpecificArgs {
                payloads,
                image_grid_thw: None,
                video_grid_thw: None,
                audio_seqlens,
            }),
            paged_attn_meta,
            flash_meta,
        });
        Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })
    }
}
