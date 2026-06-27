#![allow(clippy::cast_possible_truncation)]

//! Input processor for Qwen3-Omni serving.
//!
//! Turns a chat request into `(input_ids, OmniSpecificArgs)` for [`super::Qwen3OmniModel`]'s
//! [`crate::pipeline::MultimodalModel`] forward. **Audio** and **image** are wired end-to-end: each
//! `<|AUDIO|>` / `<|IMAGE|>` placeholder the chat template emits is expanded to the right number of
//! Thinker tokens, and the raw modality is turned into the features the validated towers consume —
//! the Whisper-style log-mel for audio (reusing [`Qwen3AsrAudioProcessor`]) and the Qwen3-VL patch
//! layout for images (reusing [`Qwen3VLImageProcessor`], so the patch math lives in exactly one
//! place). Both flow through as [`super::OmniSpecificArgs`] payloads, with the image patch grids also
//! carried separately to drive 3D mRoPE ([`super::omni_get_rope_index`]).
//!
//! The two-phase contract mirrors the established audio/vision processors: the first (scheduling) pass
//! rewrites the prompt tokens — expanding placeholders so the KV cache is sized correctly — and the
//! second (prefill) pass materializes the mel / pixel payloads. Text-only requests fall straight
//! through to the standard text input path, preserving validated text serving.
//!
//! Video frame extraction (sampling raw video into frames + the patch layout) is the remaining gap;
//! image and audio are complete. A video placeholder is reported rather than silently mis-sized.

use std::{any::Any, sync::Arc};

use anyhow::Result;
use hanzo_ml::{Device, Tensor};
use image::DynamicImage;
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
    vision_models::{
        image_processor::ImagePreProcessor, preprocessor_config::PreProcessorConfig,
        qwen3_vl::inputs_processor::Qwen3VLImageProcessor, ModelInputs,
    },
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
    image_token_id: u32,
    video_token_id: u32,
    spatial_merge_size: usize,
    audio_config: OmniAudioConfig,
    preprocessor_config: PreProcessorConfig,
}

impl Qwen3OmniProcessor {
    pub fn new(
        audio_token_id: u32,
        image_token_id: u32,
        video_token_id: u32,
        spatial_merge_size: usize,
        audio_config: OmniAudioConfig,
        preprocessor_config: PreProcessorConfig,
    ) -> Self {
        Self {
            audio_token_id,
            image_token_id,
            video_token_id,
            spatial_merge_size,
            audio_config,
            preprocessor_config,
        }
    }
}

impl Processor for Qwen3OmniProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Qwen3OmniInputsProcessor {
            audio_token_id: self.audio_token_id,
            image_token_id: self.image_token_id,
            video_token_id: self.video_token_id,
            spatial_merge_size: self.spatial_merge_size,
            audio: build_audio_processor(&self.audio_config),
            image: Qwen3VLImageProcessor::new(None),
            preprocessor_config: self.preprocessor_config.clone(),
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
    image_token_id: u32,
    video_token_id: u32,
    spatial_merge_size: usize,
    audio: Qwen3AsrAudioProcessor,
    image: Qwen3VLImageProcessor,
    preprocessor_config: PreProcessorConfig,
}

impl Qwen3OmniInputsProcessor {
    /// Patchify one image into `([num_patches, in_chans*temporal*patch*patch], [t, h, w])` via the
    /// shared Qwen3-VL image processor (one source of truth for the patch layout).
    fn preprocess_image(&self, image: DynamicImage, device: &Device) -> Result<(Tensor, [u32; 3])> {
        let pre = self
            .image
            .preprocess(
                vec![image],
                vec![],
                &self.preprocessor_config,
                device,
                (1, 1),
            )
            .map_err(anyhow::Error::msg)?;
        let grid = pre
            .image_grid_thw
            .ok_or_else(|| anyhow::Error::msg("image preprocess returned no grid_thw"))?;
        let rows = grid
            .to_dtype(hanzo_ml::DType::U32)
            .and_then(|g| g.to_vec2::<u32>())
            .map_err(anyhow::Error::msg)?;
        let r = rows
            .first()
            .ok_or_else(|| anyhow::Error::msg("empty image grid_thw"))?;
        Ok((pre.pixel_values, [r[0], r[1], r[2]]))
    }

    /// Merged Thinker-token count for a `[t, h, w]` patch grid: `t*h*w / spatial_merge_size^2`.
    fn merged_tokens(&self, grid: [u32; 3]) -> usize {
        (grid[0] as usize * grid[1] as usize * grid[2] as usize)
            / self.spatial_merge_size.pow(2).max(1)
    }
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
            return Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ));
        }
        if no_kv_cache {
            return Err(anyhow::Error::msg("Vision model must have kv cache."));
        }
        if tokenizer.is_none() {
            return Err(anyhow::Error::msg(
                "Qwen3OmniInputsProcessor requires a specified tokenizer.",
            ));
        }

        // ── Audio + image expansion / materialization (the Omni differentiator) ──────────────────
        // Phase 1 (scheduling): expand each `<|AUDIO|>` / `<|IMAGE|>` placeholder to its Thinker-token
        // length so the KV cache is sized correctly. Phase 2 (prefill): materialize the mel / pixel
        // payloads and the per-image patch grids (the latter drive 3D mRoPE in the serving forward).
        let mut payloads: Vec<(u32, ModalityInput)> = Vec::new();
        let mut audio_seqlens: Vec<usize> = Vec::new();
        let mut image_grid_rows: Vec<[u32; 3]> = Vec::new();
        if is_prompt {
            for seq in input_seqs.iter_mut() {
                let has_mm = seq.has_audios() || seq.has_images();
                if !seq.multimodal.has_changed_prompt {
                    if !has_mm {
                        continue;
                    }
                    if seq.has_videos() {
                        return Err(anyhow::Error::msg(
                            "Qwen3-Omni video input is not yet wired in the input processor \
                             (image + audio are); video frame extraction is the remaining gap.",
                        ));
                    }
                    let audios = seq.multimodal.clone_audios().unwrap_or_default();
                    let images = seq.clone_images().unwrap_or_default();
                    let toks = seq.get_toks().to_vec();
                    let mut new_toks = Vec::with_capacity(toks.len());
                    let (mut ai, mut ii) = (0usize, 0usize);
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
                        } else if t == self.image_token_id {
                            let image = images.get(ii).ok_or_else(|| {
                                anyhow::Error::msg(
                                    "more <|IMAGE|> placeholders than provided images",
                                )
                            })?;
                            let (_pixels, grid) = self.preprocess_image(image.clone(), device)?;
                            let len = self.merged_tokens(grid);
                            new_toks.extend(std::iter::repeat_n(self.image_token_id, len));
                            ii += 1;
                        } else {
                            new_toks.push(t);
                        }
                    }
                    seq.set_toks_and_reallocate(new_toks, paged_attn_metadata.as_mut());
                    seq.multimodal.has_changed_prompt = true;
                } else {
                    // Phase 2 (prefill): materialize the payloads consumed by `fuse_modalities`.
                    if let Some(audios) = seq.take_audios() {
                        for audio in &audios {
                            let mel = self
                                .audio
                                .process(audio, device)
                                .map_err(anyhow::Error::msg)?;
                            audio_seqlens.push(mel.dims()[2]);
                            payloads.push((self.audio_token_id, ModalityInput::Audio(mel)));
                        }
                    }
                    if let Some(images) = seq.take_images() {
                        for image in images {
                            let (pixels, grid) = self.preprocess_image(image, device)?;
                            image_grid_rows.push(grid);
                            let grid_thw =
                                Tensor::new(&[grid], device).map_err(anyhow::Error::msg)?;
                            payloads.push((
                                self.image_token_id,
                                ModalityInput::Image { pixels, grid_thw },
                            ));
                        }
                    }
                }
            }
        }
        let _ = self.video_token_id;

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
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
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
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks())
                    .collect::<Vec<_>>(),
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

        let image_grid_thw = if image_grid_rows.is_empty() {
            None
        } else {
            let n = image_grid_rows.len();
            let flat: Vec<u32> = image_grid_rows.iter().flatten().copied().collect();
            Some(Tensor::from_vec(flat, (n, 3), device).map_err(anyhow::Error::msg)?)
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(OmniSpecificArgs {
                payloads,
                image_grid_thw,
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
