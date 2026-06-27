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
//! Video accepts pre-extracted frames (`VideoInput::frames`, decoded upstream — mp4 → frames needs
//! FFmpeg, the caller's responsibility): each `<|VIDEO|>` placeholder is expanded to its merged-token
//! count and the frames are patchified through the same Qwen3-VL video path (frames grouped by
//! `temporal_patch_size`), flowing through as a [`ModalityInput::Video`] payload with a `[t, h, w]`
//! grid that also drives 3D mRoPE. Separate audio + video in one request works (non-interleaved, HF's
//! default); the interleaved `use_audio_in_video` layout (HF's time-chunked audio+video) is the one
//! remaining gap.

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

    /// Patchify one video's frames into `([num_patches, in_chans*temporal*patch*patch], [t, h, w])`
    /// via the shared Qwen3-VL video path: frames are grouped by `temporal_patch_size` (so `t =
    /// ceil(num_frames / temporal_patch_size)`), reusing the exact patch math the image path uses. The
    /// frames are pre-extracted upstream (mp4 → frames needs FFmpeg); this stage only patchifies them.
    fn preprocess_video(
        &self,
        frames: Vec<DynamicImage>,
        device: &Device,
    ) -> Result<(Tensor, [u32; 3])> {
        if frames.is_empty() {
            return Err(anyhow::Error::msg("video input has no frames"));
        }
        let pre = self
            .image
            .preprocess(
                vec![],
                vec![frames],
                &self.preprocessor_config,
                device,
                (1, 1),
            )
            .map_err(anyhow::Error::msg)?;
        let grid = pre
            .video_grid_thw
            .ok_or_else(|| anyhow::Error::msg("video preprocess returned no grid_thw"))?;
        let rows = grid
            .to_dtype(hanzo_ml::DType::U32)
            .and_then(|g| g.to_vec2::<u32>())
            .map_err(anyhow::Error::msg)?;
        let r = rows
            .first()
            .ok_or_else(|| anyhow::Error::msg("empty video grid_thw"))?;
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
        let mut video_grid_rows: Vec<[u32; 3]> = Vec::new();
        if is_prompt {
            for seq in input_seqs.iter_mut() {
                let has_mm = seq.has_audios() || seq.has_images() || seq.has_videos();
                if !seq.multimodal.has_changed_prompt {
                    if !has_mm {
                        continue;
                    }
                    let audios = seq.multimodal.clone_audios().unwrap_or_default();
                    let images = seq.clone_images().unwrap_or_default();
                    let videos = seq.clone_videos().unwrap_or_default();
                    let toks = seq.get_toks().to_vec();
                    let mut new_toks = Vec::with_capacity(toks.len());
                    let (mut ai, mut ii, mut vi) = (0usize, 0usize, 0usize);
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
                        } else if t == self.video_token_id {
                            let video = videos.get(vi).ok_or_else(|| {
                                anyhow::Error::msg(
                                    "more <|VIDEO|> placeholders than provided videos",
                                )
                            })?;
                            let (_pixels, grid) =
                                self.preprocess_video(video.frames.clone(), device)?;
                            let len = self.merged_tokens(grid);
                            new_toks.extend(std::iter::repeat_n(self.video_token_id, len));
                            vi += 1;
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
                    if let Some(videos) = seq.take_videos() {
                        for video in videos {
                            let (pixels, grid) = self.preprocess_video(video.frames, device)?;
                            video_grid_rows.push(grid);
                            let grid_thw =
                                Tensor::new(&[grid], device).map_err(anyhow::Error::msg)?;
                            payloads.push((
                                self.video_token_id,
                                ModalityInput::Video { pixels, grid_thw },
                            ));
                        }
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

        let stack_grid = |rows: &[[u32; 3]]| -> Result<Option<Tensor>> {
            if rows.is_empty() {
                return Ok(None);
            }
            let flat: Vec<u32> = rows.iter().flatten().copied().collect();
            Ok(Some(
                Tensor::from_vec(flat, (rows.len(), 3), device).map_err(anyhow::Error::msg)?,
            ))
        };
        let image_grid_thw = stack_grid(&image_grid_rows)?;
        let video_grid_thw = stack_grid(&video_grid_rows)?;

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(OmniSpecificArgs {
                payloads,
                image_grid_thw,
                video_grid_thw,
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

#[cfg(test)]
mod tests {
    use super::{build_audio_processor, Qwen3OmniInputsProcessor};
    use crate::vision_models::preprocessor_config::PreProcessorConfig;
    use crate::vision_models::qwen3_omni::config::OmniAudioConfig;
    use crate::vision_models::qwen3_vl::inputs_processor::Qwen3VLImageProcessor;
    use hanzo_ml::Device;
    use image::{DynamicImage, Rgb, RgbImage};

    const PATCH: usize = 16;
    const MERGE: usize = 2;
    const TEMPORAL: usize = 2;

    /// A processor with a fixed patch geometry and pixel bounds chosen so a 64×64 frame (a multiple of
    /// `patch*merge = 32`, 4096 px) is neither up- nor down-scaled — yielding a fully deterministic
    /// grid. The audio tower is unused by these shape tests (the processor only stores it).
    #[allow(clippy::field_reassign_with_default)]
    fn processor() -> Qwen3OmniInputsProcessor {
        let audio_config: OmniAudioConfig = serde_json::from_str(
            r#"{"d_model":64,"encoder_layers":1,"encoder_attention_heads":1,"encoder_ffn_dim":64,"num_mel_bins":128,"output_dim":64}"#,
        )
        .unwrap();
        let mut pc = PreProcessorConfig::default();
        pc.patch_size = Some(PATCH);
        pc.merge_size = Some(MERGE);
        pc.temporal_patch_size = Some(TEMPORAL);
        pc.min_pixels = Some(32 * 32);
        pc.max_pixels = Some(128 * 128);
        Qwen3OmniInputsProcessor {
            audio_token_id: 151646,
            image_token_id: 151655,
            video_token_id: 151656,
            spatial_merge_size: MERGE,
            audio: build_audio_processor(&audio_config),
            image: Qwen3VLImageProcessor::new(None),
            preprocessor_config: pc,
        }
    }

    /// Deterministic w×h RGB gradient (exact pixel values are irrelevant to the shape contract).
    fn synthetic(w: u32, h: u32) -> DynamicImage {
        DynamicImage::ImageRgb8(RgbImage::from_fn(w, h, |x, y| {
            Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
        }))
    }

    /// The feature width one patch row carries: `in_chans(3) * temporal_patch * patch * patch`, the
    /// exact layout [`super::super::vision::OmniVisionTower::forward`] consumes.
    fn feat() -> usize {
        3 * TEMPORAL * PATCH * PATCH
    }

    /// The image path end-to-end at the shape level: a synthetic 64×64 image patchifies to the exact
    /// `[1, 4, 4]` grid (64/16 = 4 patches per side, temporal grid 1 for a single image), the pixel
    /// rows equal the patch product, and the merged-token count phase-1 expands the `<|IMAGE|>`
    /// placeholder to equals `product / merge^2` and `rows / merge^2`. No weights, fast.
    #[test]
    fn image_path_shapes() {
        let p = processor();
        let (pixels, grid) = p.preprocess_image(synthetic(64, 64), &Device::Cpu).unwrap();
        assert_eq!(grid, [1, 4, 4]);
        assert_eq!(pixels.dims(), &[16, feat()]);
        let merged = p.merged_tokens(grid);
        let product = (grid[0] * grid[1] * grid[2]) as usize;
        assert_eq!(merged, product / (MERGE * MERGE));
        assert_eq!(merged, pixels.dims()[0] / (MERGE * MERGE));
        assert_eq!(merged, 4);
    }

    /// The video path end-to-end at the shape level: 4 pre-extracted 64×64 frames group by
    /// `temporal_patch_size` into a temporal grid of 2 (`[2, 4, 4]`), the pixel rows equal the patch
    /// product, and the `<|VIDEO|>` placeholder expands to the matching merged-token count. Mirrors
    /// `image_path_shapes`; no weights, fast.
    #[test]
    fn video_path_shapes() {
        let p = processor();
        let frames: Vec<DynamicImage> = (0..4).map(|_| synthetic(64, 64)).collect();
        let (pixels, grid) = p.preprocess_video(frames, &Device::Cpu).unwrap();
        assert_eq!(grid, [2, 4, 4]);
        assert_eq!(pixels.dims(), &[32, feat()]);
        let merged = p.merged_tokens(grid);
        let product = (grid[0] * grid[1] * grid[2]) as usize;
        assert_eq!(merged, product / (MERGE * MERGE));
        assert_eq!(merged, pixels.dims()[0] / (MERGE * MERGE));
        assert_eq!(merged, 8);
    }

    /// An odd frame count pads up to a multiple of `temporal_patch_size` (HF behaviour): 3 frames ->
    /// temporal grid 2, the same as 4 frames. Locks the temporal grouping the mRoPE positions depend on.
    #[test]
    fn video_temporal_padding() {
        let p = processor();
        let frames: Vec<DynamicImage> = (0..3).map(|_| synthetic(64, 64)).collect();
        let (pixels, grid) = p.preprocess_video(frames, &Device::Cpu).unwrap();
        assert_eq!(grid[0], 2, "3 frames pad to temporal grid 2");
        assert_eq!(pixels.dims()[0], (grid[0] * grid[1] * grid[2]) as usize);
    }

    /// Empty frames is a clean error, not a panic.
    #[test]
    fn video_empty_frames_errors() {
        let p = processor();
        assert!(p.preprocess_video(vec![], &Device::Cpu).is_err());
    }
}
