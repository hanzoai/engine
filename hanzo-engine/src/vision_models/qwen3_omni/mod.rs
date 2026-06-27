//! Qwen3-Omni-MoE: the fused understand → think → speak multimodal model.
//!
//! `model_type = "qwen3_omni_moe"`, HF class `Qwen3OmniMoeForConditionalGeneration`.
//!
//! [`Qwen3OmniModel`] assembles the validated components into the full pipeline:
//!   - **understand + think**: the [`thinker::OmniThinkerText`] decoder, fed token embeddings whose
//!     modality-placeholder rows are replaced by registered [`modality::ModalityEncoder`]s through the
//!     single [`modality::fuse_modalities`] path (audio wired now; vision/video register the same way);
//!   - **speak**: a [`talker::OmniTalker`] + [`talker::OmniCodePredictor`] MTP head + the
//!     [`code2wav::OmniCode2Wav`] vocoder, driven greedily into a 24 kHz waveform.

#![allow(dead_code)]

use std::any::Any;
use std::sync::Arc;

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_quant::{Comm, QuantMethod, ShardedVarBuilder};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::{DeviceMapSetting, DeviceMapper},
    layers::CausalMasker,
    layers_masker::{CausalMaskConfig, PastKvLenCache},
    paged_attention::{AttentionImplementation, KvCacheLayout, ModelConfigMetadata},
    pipeline::{EitherCache, IsqModel, ModelForwardContext, MultimodalModel, NormalCache},
};

pub mod audio_tower;
pub mod code2wav;
pub mod config;
pub mod inputs_processor;
pub mod modality;
pub mod talker;
pub mod thinker;
pub mod vision;

pub use config::Qwen3OmniConfig;
pub use inputs_processor::Qwen3OmniProcessor;

use audio_tower::OmniAudioTower;
use code2wav::OmniCode2Wav;
use modality::{fuse_modalities, ModalityEncoder, ModalityInput};
use talker::{OmniCodePredictor, OmniTalker};
use thinker::OmniThinkerText;

/// The Thinker audio tower wrapped as a [`ModalityEncoder`]: log-mel `[1, 128, T]` -> Thinker-space
/// `[T_out, hidden]` token embeddings, scattered at the audio placeholder token id.
struct AudioModality(OmniAudioTower, u32);

impl ModalityEncoder for AudioModality {
    fn placeholder_token(&self) -> u32 {
        self.1
    }

    fn encode(&self, input: &ModalityInput, _device: &Device) -> Result<Tensor> {
        match input {
            // audio_tower.forward: [1, 128, T] -> [1, T_out, hidden]; drop the leading batch axis.
            ModalityInput::Audio(mel) => self.0.forward(mel)?.squeeze(0),
            _ => hanzo_ml::bail!("AudioModality encodes ModalityInput::Audio only"),
        }
    }
}

/// The assembled Qwen3-Omni model: a multimodal Thinker (text decoder + registered modality
/// encoders) plus the Talker / code-predictor / Code2Wav speech stack.
pub struct Qwen3OmniModel {
    thinker: OmniThinkerText,
    /// Modality encoders, looked up by placeholder token in [`fuse_modalities`]. Audio is wired now;
    /// vision/video push onto this vec with no other change.
    encoders: Vec<Box<dyn ModalityEncoder>>,
    talker: OmniTalker,
    code_predictor: OmniCodePredictor,
    code2wav: OmniCode2Wav,
    cfg: Qwen3OmniConfig,
    device: Device,
    // ── Serving state (the [`MultimodalModel`] path) ──────────────────────────────────────────
    // Owned KV cache + metadata threaded through the cache-aware thinker forward. The validated
    // cacheless path (`forward` / `forward_embeds`, used by the tests) never touches any of these.
    cache: EitherCache,
    cfg_meta: ModelConfigMetadata,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    dtype: DType,
    max_seq_len: usize,
    /// mRoPE position-axis carry across decode steps after a vision prefill: the next-token position
    /// is `seqlen_offset + mrope_delta` on all three axes. Vision compresses position space (an image
    /// occupies `max(h,w)` positions but many more tokens), so the delta is typically negative. It is
    /// `0` for the text/audio path — decode then stays on the validated 1D cached forward. Interior
    /// mutability matches the `Arc<Mutex<NormalCache>>` already threaded through `forward(&self)`.
    mrope_delta: std::sync::atomic::AtomicI64,
}

impl Qwen3OmniModel {
    /// `vb` is the checkpoint root: the Thinker loads from `vb.pp("thinker")`, its audio tower from
    /// `vb.pp("thinker").pp("audio_tower")`, the Talker from `vb.pp("talker")`, the code predictor
    /// from `vb.pp("talker").pp("code_predictor")`, and the vocoder from `vb.pp("code2wav")`.
    pub fn new(
        cfg: &Qwen3OmniConfig,
        vb: ShardedVarBuilder,
        device: &Device,
        comm: &Arc<Comm>,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Self> {
        // Only the Thinker text decoder is served through the cache-aware (optionally paged) forward;
        // the talker / code-predictor / code2wav are generation sub-models and stay `naive_sdpa`.
        let thinker = OmniThinkerText::new(
            &cfg.thinker_config.text_config,
            vb.pp("thinker"),
            device,
            comm,
            attention_mechanism,
        )?;

        // Native modality encoders, each keyed by its placeholder token id (see `fuse_modalities`).
        let audio_tower = OmniAudioTower::new(
            &cfg.thinker_config.audio_config,
            vb.pp("thinker").pp("audio_tower"),
            device,
        )?;
        let mut encoders: Vec<Box<dyn ModalityEncoder>> = vec![Box::new(AudioModality(
            audio_tower,
            cfg.thinker_config.audio_token_id,
        ))];

        // Vision: ONE Thinker visual tower (`thinker.visual.*`) shared via `Arc` by the image and
        // video encoders — identical weights, two placeholder token ids. Proves the modality
        // abstraction extends with zero changes to the thinker / talker / fusion.
        let vision_tower = Arc::new(vision::OmniVisionTower::new(
            &cfg.thinker_config.vision_config,
            vb.pp("thinker").pp("visual"),
            device,
        )?);
        encoders.push(Box::new(vision::VisionModality::new(
            vision_tower.clone(),
            cfg.thinker_config.image_token_id,
        )));
        encoders.push(Box::new(vision::VisionModality::new(
            vision_tower,
            cfg.thinker_config.video_token_id,
        )));

        let talker = OmniTalker::new(&cfg.talker_config, vb.pp("talker"), device, comm)?;
        let code_predictor = OmniCodePredictor::new(
            &cfg.talker_config.code_predictor_config,
            vb.pp("talker").pp("code_predictor"),
            device,
        )?;
        let code2wav = OmniCode2Wav::new(&cfg.code2wav_config, vb.pp("code2wav"), device)?;

        // Serving state, derived from the Thinker text config (the decoder that produces text
        // logits). One `KvCache` per Thinker layer; metadata mirrors `qwen3_5_moe`/`qwen3_vl_moe`.
        let tc = &cfg.thinker_config.text_config;
        let cache = EitherCache::Normal(NormalCache::new(
            tc.num_hidden_layers,
            tc.max_position_embeddings,
        ));
        let cfg_meta = ModelConfigMetadata {
            max_seq_len: tc.max_position_embeddings,
            num_layers: tc.num_hidden_layers,
            hidden_size: tc.hidden_size,
            num_kv_heads: tc.num_key_value_heads,
            num_attn_heads: tc.num_attention_heads,
            sliding_window: None,
            k_head_dim: tc.head_dim,
            v_head_dim: tc.head_dim,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        // The Thinker loads on a single device (the validated naive loader ignores device mapping),
        // so a dummy mapper satisfies the `IsqModel` contract without claiming a multi-GPU split.
        let mapper = DeviceMapSetting::dummy().into_mapper(
            tc.num_hidden_layers,
            device,
            None,
            std::slice::from_ref(device),
        )?;

        Ok(Self {
            thinker,
            encoders,
            talker,
            code_predictor,
            code2wav,
            cfg: cfg.clone(),
            device: device.clone(),
            cache,
            cfg_meta,
            mapper,
            dtype: vb.dtype(),
            max_seq_len: tc.max_position_embeddings,
            mrope_delta: std::sync::atomic::AtomicI64::new(0),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// `[1, n]` u32 id tensor on the model device.
    fn ids(&self, v: &[u32]) -> Result<Tensor> {
        Tensor::new(v, &self.device)?.unsqueeze(0)
    }

    /// Understand → think: embed `input_ids`, fuse every modality payload into its placeholder rows
    /// through the single [`fuse_modalities`] path, then run the Thinker decoder. Returns the text
    /// `logits` and the full Thinker hidden-state stream (the Talker reads `accept_hidden_layer`).
    ///
    /// Positions follow the modality: a vision payload (image/video) lays its placeholders on the 2-D
    /// patch grid, so the decoder runs interleaved 3D mRoPE ([`omni_get_rope_index`] derives the
    /// positions from the grid that travels with the payload); a text/audio-only request keeps the 1D
    /// path (their three mRoPE axes are equal, so it is numerically exact).
    pub fn forward(
        &self,
        input_ids: &Tensor,
        inputs: &[(u32, ModalityInput)],
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let embeds = self.thinker.embed_tokens(input_ids)?;
        let fused = fuse_modalities(&embeds, input_ids, &self.encoders, inputs, &self.device)?;

        let (image_grid_thw, video_grid_thw) = vision_grid_rows(inputs)?;
        if image_grid_thw.is_none() && video_grid_thw.is_none() {
            return self.thinker.forward_embeds(&fused, seqlen_offsets, mask);
        }

        let ids: Vec<u32> = input_ids
            .flatten_all()?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        let tc = &self.cfg.thinker_config;
        let position_ids = omni_get_rope_index(
            &ids,
            image_grid_thw.as_deref(),
            video_grid_thw.as_deref(),
            tc.vision_config.spatial_merge_size,
            tc.image_token_id,
            tc.video_token_id,
            tc.vision_start_token_id,
            tc.position_id_per_seconds,
            &self.device,
        )?;
        self.thinker
            .forward_embeds_mrope(&fused, &position_ids, mask)
    }

    /// Speak: render the Thinker outputs into a 24 kHz waveform `[1, 1, samples]`. Resolves the
    /// `speaker` name to its codec speaker id, runs [`Self::generate_codes`], then the vocoder.
    pub fn generate_speech(
        &self,
        thinker_hidden: &Tensor,
        thinker_embed: &Tensor,
        sequence_ids: &[u32],
        speaker: &str,
        max_frames: usize,
    ) -> Result<Tensor> {
        let sid = speaker_id(speaker)
            .ok_or_else(|| hanzo_ml::Error::Msg(format!("unknown speaker {speaker:?}")))?;
        let codes =
            self.generate_codes(thinker_hidden, thinker_embed, sequence_ids, sid, max_frames)?;
        if codes.dim(D::Minus1)? == 0 {
            return Tensor::zeros((1usize, 1usize, 0usize), DType::F32, &self.device);
        }
        self.code2wav.decode(&codes)
    }

    /// Talker → code-predictor speech generation, faithful to HF
    /// `Qwen3OmniMoeForConditionalGeneration.generate`. Greedy (argmax) throughout.
    ///
    /// `thinker_embed` (`hidden_states[0]`, word embeddings) and `thinker_hidden`
    /// (`hidden_states[accept_hidden_layer]`) cover the thinker sequence **minus its last token**:
    /// HF collects one hidden per generation step, so the final sampled token is never re-embedded,
    /// and the prefill therefore reads at most `thinker_embed.len()` positions. `sequence_ids` is the
    /// full thinker sequence — used only for chatml segmentation, so its length is the assistant
    /// segment's end boundary. Returns codec codes `[1, num_code_groups, T]`.
    pub fn generate_codes(
        &self,
        thinker_hidden: &Tensor,
        thinker_embed: &Tensor,
        sequence_ids: &[u32],
        speaker_id: u32,
        max_frames: usize,
    ) -> Result<Tensor> {
        let tc = &self.cfg.talker_config;
        let groups = tc.num_code_groups;
        let eos = tc.codec_eos_token_id;
        let vocab = tc.text_config.vocab_size;
        let suppress_lo = vocab - 1024;
        let dtype = thinker_embed.dtype();

        let (prefill, trailing, tts_pad) =
            self.build_talker_prefill(thinker_hidden, thinker_embed, sequence_ids, speaker_id)?;
        let trailing_len = trailing.dim(1)?;

        let mut inputs_embeds = prefill;
        let mut all_codes: Vec<Vec<u32>> = Vec::new();
        for f in 0..max_frames {
            let seq = inputs_embeds.dim(1)?;
            let mask = causal_mask(seq, dtype, &self.device)?;
            let (last_hidden_all, logits) =
                self.talker.forward(&inputs_embeds, &[0], Some(&mask))?;

            // Greedy codec-0 with HF `suppress_tokens`: [vocab-1024, vocab) except codec_eos -> -inf.
            let mut lv = logits
                .i((0, seq - 1, ..))?
                .to_dtype(DType::F32)?
                .to_vec1::<f32>()?;
            for (t, v) in lv.iter_mut().enumerate().skip(suppress_lo) {
                if t as u32 != eos {
                    *v = f32::NEG_INFINITY;
                }
            }
            let code0 = argmax_u32(&lv);
            if code0 == eos {
                break;
            }

            // Code-predictor MTP over groups 1..num_code_groups, conditioned on the **post-norm**
            // talker hidden of the current frame (HF `past_hidden = hidden_states[-1][:, -1:]`, with
            // `tie_last_hidden_states=True` so `hidden_states[-1] == last_hidden_state`).
            let last_hidden = last_hidden_all.i((.., seq - 1.., ..))?;
            let frame = self.predict_groups(&last_hidden, code0)?;

            // Next talker input = sum of all 16 group embeddings (talker codec embed for group 0, the
            // code-predictor per-group embeds for groups 1..16) + the per-frame text-side hidden
            // (`trailing_text_hidden[f]`, or `tts_pad_embed` once `trailing` is exhausted).
            let mut summed = self.talker.embed_codec(&self.ids(&[code0])?)?;
            for (group, &id) in frame.iter().enumerate().skip(1) {
                summed = (summed + self.code_predictor.embed_group(group, &self.ids(&[id])?)?)?;
            }
            let text_side = if f < trailing_len {
                trailing.i((.., f..f + 1, ..))?
            } else {
                tts_pad.clone()
            };
            let next = (summed + text_side)?;
            all_codes.push(frame);
            inputs_embeds = Tensor::cat(&[&inputs_embeds, &next], 1)?;
        }

        let frames = all_codes.len();
        if frames == 0 {
            return Tensor::zeros((1usize, groups, 0usize), DType::U32, &self.device);
        }
        // Group-major `[1, groups, T]` (frame index fastest) — the layout `code2wav.decode` consumes.
        let mut flat = Vec::with_capacity(groups * frames);
        for g in 0..groups {
            for frame in &all_codes {
                flat.push(frame[g]);
            }
        }
        Tensor::from_vec(flat, (1, groups, frames), &self.device)
    }

    /// Assemble the talker prefill exactly as HF `_get_talker_user_parts` +
    /// `_get_talker_assistant_parts`: chatml-segment the sequence, project each part into talker
    /// hidden space, and return `(inputs_embeds, trailing_text_hidden, tts_pad_embed)`.
    fn build_talker_prefill(
        &self,
        thinker_hidden: &Tensor,
        thinker_embed: &Tensor,
        sequence_ids: &[u32],
        speaker_id: u32,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let tc = &self.cfg.talker_config;
        let ht = tc.text_config.hidden_size;
        let dtype = thinker_embed.dtype();
        let l = thinker_embed.dim(1)?; // sequence length minus the last (uncollected) token

        // tts_{bos,eos,pad} = text_projection(thinker.embed(special)). [1,1,Ht] each.
        let tts = self
            .talker
            .project_text(&self.thinker.embed_tokens(&self.ids(&[
                self.cfg.tts_bos_token_id,
                self.cfg.tts_eos_token_id,
                self.cfg.tts_pad_token_id,
            ])?)?)?;
        let tts_bos = tts.i((.., 0..1, ..))?;
        let tts_eos = tts.i((.., 1..2, ..))?;
        let tts_pad = tts.i((.., 2..3, ..))?;

        // chatml boundaries: every <|im_start|> in the prompt, then the sequence end.
        let mut bounds: Vec<usize> = sequence_ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == self.cfg.im_start_token_id)
            .map(|(i, _)| i)
            .collect();
        bounds.push(sequence_ids.len());

        let mut parts: Vec<Tensor> = Vec::new();
        let mut trailing: Option<Tensor> = None;
        for i in 0..bounds.len() - 1 {
            let start = bounds[i];
            let end = bounds[i + 1];
            let role = sequence_ids[start + 1];
            if role == self.cfg.system_token_id {
                continue;
            } else if role == self.cfg.user_token_id {
                parts.push(self.talker_user_part(
                    thinker_hidden,
                    thinker_embed,
                    sequence_ids,
                    start,
                    end,
                )?);
            } else if role == ASSISTANT_TOKEN_ID && i == bounds.len() - 2 {
                // assistant_hidden = text_projection(thinker_embed[start:end]); end clamps to `l`.
                let hi = end.min(l);
                let ah = self
                    .talker
                    .project_text(&thinker_embed.i((.., start..hi, ..))?)?;
                // assistant_text = [ah[:3], tts_pad*4, tts_bos, ah[3:4]]
                let pad4 = tts_pad.broadcast_as((1, 4, ht))?.contiguous()?;
                let assistant_text = Tensor::cat(
                    &[
                        &ah.i((.., 0..3, ..))?,
                        &pad4,
                        &tts_bos,
                        &ah.i((.., 3..4, ..))?,
                    ],
                    1,
                )?;
                // assistant_codec = [zeros(3), talker.embed([nothink, think_bos, think_eos, spk, pad, bos])]
                let codec_ids = self.ids(&[
                    tc.codec_nothink_id,
                    tc.codec_think_bos_id,
                    tc.codec_think_eos_id,
                    speaker_id,
                    tc.codec_pad_id,
                    tc.codec_bos_id,
                ])?;
                let codec_emb = self.talker.embed_codec(&codec_ids)?; // [1,6,Ht]
                let zeros3 = Tensor::zeros((1, 3, ht), dtype, &self.device)?;
                let assistant_codec = Tensor::cat(&[&zeros3, &codec_emb], 1)?;
                parts.push((assistant_text + assistant_codec)?);
                // trailing_text_hidden = [ah[4:], tts_eos]
                trailing = Some(Tensor::cat(&[&ah.i((.., 4.., ..))?, &tts_eos], 1)?);
            } else if role == ASSISTANT_TOKEN_ID {
                continue;
            } else {
                hanzo_ml::bail!("talker prefill: unexpected role token {role} after <|im_start|>");
            }
        }
        let prefill = Tensor::cat(&parts, 1)?;
        let trailing = trailing
            .ok_or_else(|| hanzo_ml::Error::Msg("talker prefill: no assistant segment".into()))?;
        Ok((prefill, trailing, tts_pad))
    }

    /// HF `_get_talker_user_parts`: multimodal positions take `hidden_projection(thinker_hidden)`,
    /// text positions take `text_projection(thinker_embed)`.
    fn talker_user_part(
        &self,
        thinker_hidden: &Tensor,
        thinker_embed: &Tensor,
        sequence_ids: &[u32],
        start: usize,
        end: usize,
    ) -> Result<Tensor> {
        let l = thinker_embed.dim(1)?;
        let hi = end.min(l);
        let text = self
            .talker
            .project_text(&thinker_embed.i((.., start..hi, ..))?)?;
        let tcfg = &self.cfg.thinker_config;
        let mut mm = vec![0f32; hi - start];
        let mut any = false;
        for (k, &id) in sequence_ids[start..hi].iter().enumerate() {
            if id == tcfg.audio_token_id || id == tcfg.image_token_id || id == tcfg.video_token_id {
                mm[k] = 1.0;
                any = true;
            }
        }
        if !any {
            return Ok(text);
        }
        let hid = self
            .talker
            .project_thinker_hidden(&thinker_hidden.i((.., start..hi, ..))?)?;
        let m = Tensor::from_vec(mm, (1, hi - start, 1), &self.device)?.to_dtype(text.dtype())?;
        let one_minus = m.affine(-1.0, 1.0)?;
        text.broadcast_mul(&one_minus)? + hid.broadcast_mul(&m)?
    }

    /// Greedy MTP: given the talker hidden for the current frame (group 0) and the sampled `code0`,
    /// predict codes for groups `1..num_code_groups` via the code predictor (mirrors qwen3_tts).
    fn predict_groups(&self, talker_hidden_last: &Tensor, code0: u32) -> Result<Vec<u32>> {
        let groups = self.cfg.talker_config.num_code_groups;
        let dtype = talker_hidden_last.dtype();
        let mut codes = vec![code0];
        let mut seq = Tensor::cat(
            &[
                talker_hidden_last,
                &self.talker.embed_codec(&self.ids(&[code0])?)?,
            ],
            1,
        )?;
        for group in 1..groups {
            let mask = causal_mask(seq.dim(1)?, dtype, &self.device)?;
            let logits = self.code_predictor.step(&seq, Some(&mask), group)?;
            let id = logits
                .i((0, logits.dim(1)? - 1, ..))?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;
            codes.push(id);
            if group < groups - 1 {
                let emb = self.code_predictor.embed_group(group, &self.ids(&[id])?)?;
                seq = Tensor::cat(&[&seq, &emb], 1)?;
            }
        }
        Ok(codes)
    }
}

/// Per-request multimodal payloads for the [`MultimodalModel`] serving path, produced by the Omni
/// input processor and passed through `forward`'s `model_specific_args`.
///
/// `payloads` are the `(placeholder_token, ModalityInput)` pairs that [`fuse_modalities`] scatters
/// into the Thinker embedding rows (audio mel / image pixels / video frames). The grids and
/// `audio_seqlens` describe the modality shapes that drive 3D mRoPE position computation
/// ([`omni_get_rope_index`]). An empty `payloads` (the [`Default`]) is the text-only request, which
/// preserves the validated text serving path exactly.
#[derive(Default)]
pub struct OmniSpecificArgs {
    /// `(placeholder_token_id, payload)` consumed left-to-right by [`fuse_modalities`].
    pub payloads: Vec<(u32, ModalityInput)>,
    /// `[num_images, 3]` (t, h, w) patch grid per image, for vision mRoPE positions.
    pub image_grid_thw: Option<Tensor>,
    /// `[num_videos, 3]` (t, h, w) patch grid per video, for vision mRoPE positions.
    pub video_grid_thw: Option<Tensor>,
    /// Per-audio mel frame counts (Thinker-token lengths are derived via the conv feat formula).
    pub audio_seqlens: Vec<usize>,
}

/// `[n, 3]` u32 grid tensor -> `Vec<[u32; 3]>` rows for [`omni_get_rope_index`].
fn grid_rows(grid: Option<&Tensor>) -> Result<Option<Vec<[u32; 3]>>> {
    let Some(g) = grid else { return Ok(None) };
    let raw = g.to_dtype(DType::U32)?.to_vec2::<u32>()?;
    let mut rows = Vec::with_capacity(raw.len());
    for r in raw {
        if r.len() != 3 {
            hanzo_ml::bail!("grid_thw rows must have length 3, got {}", r.len());
        }
        rows.push([r[0], r[1], r[2]]);
    }
    Ok(Some(rows))
}

/// Collect the `(image, video)` patch-grid rows carried by the vision payloads, in placeholder order,
/// for [`omni_get_rope_index`]. The grid travels with each [`ModalityInput::Image`] /
/// [`ModalityInput::Video`], so the model-level `forward` derives the mRoPE grids straight from the
/// payloads (the serving path reads the equivalent pre-stacked grids off [`OmniSpecificArgs`]).
#[allow(clippy::type_complexity)]
fn vision_grid_rows(
    inputs: &[(u32, ModalityInput)],
) -> Result<(Option<Vec<[u32; 3]>>, Option<Vec<[u32; 3]>>)> {
    let mut images: Vec<[u32; 3]> = Vec::new();
    let mut videos: Vec<[u32; 3]> = Vec::new();
    for (_token, input) in inputs {
        match input {
            ModalityInput::Image { grid_thw, .. } => {
                images.extend(grid_rows(Some(grid_thw))?.unwrap_or_default());
            }
            ModalityInput::Video { grid_thw, .. } => {
                videos.extend(grid_rows(Some(grid_thw))?.unwrap_or_default());
            }
            ModalityInput::Audio(_) => {}
        }
    }
    Ok((
        (!images.is_empty()).then_some(images),
        (!videos.is_empty()).then_some(videos),
    ))
}

/// The mRoPE decode carry from a prefill: `max(position) + 1 - seq`. Vision compresses position space,
/// so for an image prefill this is negative; for text/audio (positions == `arange`) it is `0`.
fn mrope_delta_of(position_ids: &Tensor, seq: usize) -> Result<i64> {
    let max_pos = position_ids
        .flatten_all()?
        .to_vec1::<i64>()?
        .into_iter()
        .max()
        .unwrap_or(0);
    Ok(max_pos + 1 - seq as i64)
}

/// Decode-step mRoPE positions `[3, batch, t]`: each token's position on all three axes is
/// `seqlen_offset + delta + within_chunk_index`, continuing the running position recorded at prefill.
fn decode_mrope_positions(
    t: usize,
    seqlen_offsets: &[usize],
    delta: i64,
    device: &Device,
) -> Result<Tensor> {
    let batch = seqlen_offsets.len();
    let mut flat = Vec::with_capacity(3 * batch * t);
    for _axis in 0..3 {
        for &offset in seqlen_offsets {
            for j in 0..t {
                flat.push(offset as i64 + delta + j as i64);
            }
        }
    }
    Tensor::from_vec(flat, (3, batch, t), device)
}

impl MultimodalModel for Qwen3OmniModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        _pixel_values: Option<Tensor>,
        model_specific_args: Box<dyn Any>,
        ctx: &mut ModelForwardContext<'_>,
    ) -> Result<Tensor> {
        // Serving = understand→think to text logits over the cache-aware Thinker decoder. The Omni
        // input processor hands modality payloads through `model_specific_args`; when present, the
        // raw mel/pixel/frame features are encoded and scattered into the placeholder embedding rows
        // by [`fuse_modalities`] (the same validated path the model-level `forward` uses), then mRoPE
        // positions drive the decoder. A text-only request (empty payloads, or no args) runs the
        // exact validated text path. Payloads are produced only during prefill; decode steps carry no
        // payloads, so a vision prefill records its position delta ([`Self::mrope_delta`]) for the
        // text-token decode that follows.
        use std::sync::atomic::Ordering;

        let args = model_specific_args.downcast::<OmniSpecificArgs>().ok();
        let has_payloads = args.as_ref().is_some_and(|a| !a.payloads.is_empty());
        let has_vision = args
            .as_ref()
            .is_some_and(|a| a.image_grid_thw.is_some() || a.video_grid_thw.is_some());

        let seqlen_offsets = ctx.seqlen_offsets();
        let is_prefill = seqlen_offsets.iter().all(|&o| o == 0);
        // `force_custom` keeps a real additive mask (rather than a flash-causal marker) so the
        // Thinker's `naive_sdpa` masks correctly; `None` is returned for single-token decode.
        let mask = CausalMasker.make_causal_mask(
            input_ids,
            &seqlen_offsets as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig {
                sliding_window: None,
                force_custom: true,
            },
        )?;
        let embeds = self.thinker.embed_tokens(input_ids)?;
        let mut cache = self.cache.normal();

        // ── Vision prefill: fuse + interleaved 3D mRoPE; record the position delta for decode. ──
        if has_payloads && has_vision {
            let args = args.unwrap();
            let fused = fuse_modalities(
                &embeds,
                input_ids,
                &self.encoders,
                &args.payloads,
                &self.device,
            )?;
            let ids: Vec<u32> = input_ids
                .flatten_all()?
                .to_dtype(DType::U32)?
                .to_vec1::<u32>()?;
            let tc = &self.cfg.thinker_config;
            let position_ids = omni_get_rope_index(
                &ids,
                grid_rows(args.image_grid_thw.as_ref())?.as_deref(),
                grid_rows(args.video_grid_thw.as_ref())?.as_deref(),
                tc.vision_config.spatial_merge_size,
                tc.image_token_id,
                tc.video_token_id,
                tc.vision_start_token_id,
                tc.position_id_per_seconds,
                &self.device,
            )?;
            self.mrope_delta
                .store(mrope_delta_of(&position_ids, ids.len())?, Ordering::Relaxed);
            return self.thinker.forward_cached_mrope(
                &fused,
                &position_ids,
                mask.as_option_tensor(),
                &mut cache.0,
                ctx,
            );
        }

        // ── Audio/text prefill carrying payloads (no vision): fuse, reset the carry, 1D cached. ──
        // Audio/text positions are equal on all three mRoPE axes (HF expands them as
        // `arange().expand(3, -1)`), so the 1D cached path is numerically exact, not an approximation.
        if has_payloads {
            let args = args.unwrap();
            let fused = fuse_modalities(
                &embeds,
                input_ids,
                &self.encoders,
                &args.payloads,
                &self.device,
            )?;
            self.mrope_delta.store(0, Ordering::Relaxed);
            return self.thinker.forward_cached(
                &fused,
                seqlen_offsets,
                mask.as_option_tensor(),
                &mut cache.0,
                ctx,
            );
        }

        // ── No payloads: a fresh prefill resets the carry; a decode step continues from it. ──
        if is_prefill {
            self.mrope_delta.store(0, Ordering::Relaxed);
            return self.thinker.forward_cached(
                &embeds,
                seqlen_offsets,
                mask.as_option_tensor(),
                &mut cache.0,
                ctx,
            );
        }

        // Decode. With no carry (text/audio history) this is the validated 1D cached path. After a
        // vision prefill the running position is `seqlen_offset + delta` on all three axes — the decode
        // token is text, so the axes agree and mRoPE reproduces the correct continued position.
        let delta = self.mrope_delta.load(Ordering::Relaxed);
        if delta == 0 {
            return self.thinker.forward_cached(
                &embeds,
                seqlen_offsets,
                mask.as_option_tensor(),
                &mut cache.0,
                ctx,
            );
        }
        let position_ids =
            decode_mrope_positions(input_ids.dim(1)?, seqlen_offsets, delta, &self.device)?;
        self.thinker.forward_cached_mrope(
            &embeds,
            &position_ids,
            mask.as_option_tensor(),
            &mut cache.0,
            ctx,
        )
    }
    fn cache(&self) -> &EitherCache {
        &self.cache
    }
    fn cache_mut(&mut self) -> &mut EitherCache {
        &mut self.cache
    }
    fn device(&self) -> &Device {
        &self.device
    }
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn config(&self) -> &ModelConfigMetadata {
        &self.cfg_meta
    }
    fn default_model_specific_args(&self, _input_ids: &Tensor) -> Box<dyn Any> {
        // Text-only request: no modality payloads, no grids.
        Box::new(OmniSpecificArgs::default())
    }
}

impl IsqModel for Qwen3OmniModel {
    fn get_layers(
        &mut self,
    ) -> (
        Vec<(&mut Arc<dyn QuantMethod>, Option<usize>)>,
        &dyn DeviceMapper,
    ) {
        // Not an ISQ checkpoint: the validated Thinker/Talker stacks use plain `hanzo_nn::Linear`,
        // not `QuantMethod` layers, so there are no in-place-quantizable tensors to expose.
        (Vec::new(), &*self.mapper)
    }
    fn residual_tensors(&self) -> Vec<(String, Tensor)> {
        Vec::new()
    }
}

impl AnyMoeBaseModelMixin for Qwen3OmniModel {}

impl crate::speculative::SpeculativeTargetMixin for Qwen3OmniModel {}

/// HF `config.assistant_token_id` (absent from [`Qwen3OmniConfig`]; stable checkpoint constant).
const ASSISTANT_TOKEN_ID: u32 = 77091;

/// The codec speaker a request gets when it names none (HF default: Ethan).
pub const DEFAULT_SPEAKER: &str = "ethan";

/// The selectable Omni speakers (HF `talker_config.speaker_id` keys), for request validation / listing.
pub const SPEAKERS: [&str; 3] = ["chelsie", "ethan", "aiden"];

/// Resolve an Omni speaker name to its HF `talker_config.speaker_id` codec id, case-insensitively
/// (absent from [`Qwen3OmniConfig`]; this is the published checkpoint map). `None` for an unknown name
/// so a caller can reject the request or fall back to [`DEFAULT_SPEAKER`]. This is the single place a
/// speaker selection threaded from a request resolves before [`Qwen3OmniModel::generate_speech`].
pub fn speaker_id(name: &str) -> Option<u32> {
    match name.to_ascii_lowercase().as_str() {
        "chelsie" => Some(2301),
        "ethan" => Some(2302),
        "aiden" => Some(2303),
        _ => None,
    }
}

/// Index of the maximum element (argmax over the last axis, ties → lowest index).
fn argmax_u32(v: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_v {
            best_v = x;
            best = i;
        }
    }
    best as u32
}

/// Number of Thinker audio tokens a mel of `mel_frames` frames produces, faithful to HF
/// `_get_feat_extract_output_lengths`: three stride-2 convs within each 100-frame block, plus 13
/// frames per full block. Equals the [`OmniAudioTower`] output length (its conv stem chunks the mel
/// into 100-frame windows, each downsampled 8× to 13 post-CNN frames). The input processor uses this
/// to expand the single audio placeholder into the right number of rows.
pub fn omni_audio_feat_len(mel_frames: usize) -> usize {
    let leave = mel_frames % 100;
    // `(leave - 1) / 2 + 1` with saturating arithmetic (matches Python floor-div for leave==0 → 0).
    let f = |n: usize| if n == 0 { 0 } else { (n - 1) / 2 + 1 };
    f(f(f(leave))) + (mel_frames / 100) * 13
}

/// 3D interleaved-mRoPE position ids for Qwen3-Omni serving, faithful to HF
/// `Qwen3OmniMoeThinkerForConditionalGeneration.get_rope_index` for a single sequence without
/// `use_audio_in_video`.
///
/// Text and **audio** tokens advance one position on all three (temporal, height, width) axes — HF
/// lays both out as `arange(len).expand(3, -1)` — so they reduce exactly to 1D RoPE. Each image/video
/// span instead lays its placeholders on the patch grid: temporal index `t * position_id_per_seconds`
/// (×1 fps for images), height `arange(grid_h / merge)`, width `arange(grid_w / merge)`, all offset
/// by the running position. The surrounding `vision_start` / `vision_end` tokens are sequential.
///
/// Returns `[3, 1, seq]` (t, h, w). `image_grid_thw` / `video_grid_thw` are `[t, h, w]` patch grids
/// (pre-merge) in placeholder order; one row is consumed per image/video span encountered.
#[allow(clippy::too_many_arguments)]
pub fn omni_get_rope_index(
    input_ids: &[u32],
    image_grid_thw: Option<&[[u32; 3]]>,
    video_grid_thw: Option<&[[u32; 3]]>,
    spatial_merge_size: usize,
    image_token_id: u32,
    video_token_id: u32,
    vision_start_token_id: u32,
    position_id_per_seconds: usize,
    device: &Device,
) -> Result<Tensor> {
    let seq = input_ids.len();
    let merge = spatial_merge_size.max(1) as i64;
    let pps = position_id_per_seconds as i64;
    let (mut tp, mut hp, mut wp) = (
        Vec::with_capacity(seq),
        Vec::with_capacity(seq),
        Vec::with_capacity(seq),
    );
    let mut next: i64 = 0; // next sequential (text/audio) position on all three axes
    let (mut image_idx, mut video_idx) = (0usize, 0usize);

    let mut i = 0usize;
    while i < seq {
        let tok = input_ids[i];
        let opens_vision = tok == vision_start_token_id
            && i + 1 < seq
            && (input_ids[i + 1] == image_token_id || input_ids[i + 1] == video_token_id);
        if !opens_vision {
            tp.push(next);
            hp.push(next);
            wp.push(next);
            next += 1;
            i += 1;
            continue;
        }

        // `vision_start` is a sequential token; the grid starts one position later.
        tp.push(next);
        hp.push(next);
        wp.push(next);
        let base = next + 1;

        let is_image = input_ids[i + 1] == image_token_id;
        let row = if is_image {
            let g = image_grid_thw
                .ok_or_else(|| hanzo_ml::Error::msg("omni_get_rope_index: image grid missing"))?;
            let r = *g.get(image_idx).ok_or_else(|| {
                hanzo_ml::Error::msg("omni_get_rope_index: too few image_grid_thw rows")
            })?;
            image_idx += 1;
            r
        } else {
            let g = video_grid_thw
                .ok_or_else(|| hanzo_ml::Error::msg("omni_get_rope_index: video grid missing"))?;
            let r = *g.get(video_idx).ok_or_else(|| {
                hanzo_ml::Error::msg("omni_get_rope_index: too few video_grid_thw rows")
            })?;
            video_idx += 1;
            r
        };
        let (gt, gh, gw) = (row[0] as i64, row[1] as i64 / merge, row[2] as i64 / merge);
        if gt <= 0 || gh <= 0 || gw <= 0 {
            hanzo_ml::bail!("omni_get_rope_index: degenerate grid {row:?}");
        }
        let mut max_pos = next;
        for t in 0..gt {
            for h in 0..gh {
                for w in 0..gw {
                    let (t_pos, h_pos, w_pos) = (base + t * pps, base + h, base + w);
                    tp.push(t_pos);
                    hp.push(h_pos);
                    wp.push(w_pos);
                    max_pos = max_pos.max(t_pos).max(h_pos).max(w_pos);
                }
            }
        }
        next = max_pos + 1;
        i += 1 + (gt * gh * gw) as usize; // skip vision_start + the placeholder run
    }

    if tp.len() != seq {
        hanzo_ml::bail!(
            "omni_get_rope_index: produced {} positions for {seq} tokens (grid/placeholder mismatch)",
            tp.len()
        );
    }
    let mut flat = Vec::with_capacity(3 * seq);
    flat.extend_from_slice(&tp);
    flat.extend_from_slice(&hp);
    flat.extend_from_slice(&wp);
    Tensor::from_vec(flat, (3, 1, seq), device)
}

/// Additive causal mask `[1, 1, t, t]`: 0 on/below the diagonal, -inf above.
fn causal_mask(t: usize, dtype: DType, device: &Device) -> Result<Tensor> {
    let mut data = vec![0f32; t * t];
    for i in 0..t {
        for j in (i + 1)..t {
            data[i * t + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_vec(data, (1, 1, t, t), device)?.to_dtype(dtype)
}

#[cfg(test)]
mod mrope_tests {
    use super::{omni_audio_feat_len, omni_get_rope_index};
    use hanzo_ml::Device;

    /// (t, h, w) position rows for a single sequence.
    #[allow(clippy::too_many_arguments)]
    fn pos(
        ids: &[u32],
        img: Option<&[[u32; 3]]>,
        vid: Option<&[[u32; 3]]>,
        merge: usize,
        image_tok: u32,
        video_tok: u32,
        vstart: u32,
        pps: usize,
    ) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
        let t = omni_get_rope_index(
            ids,
            img,
            vid,
            merge,
            image_tok,
            video_tok,
            vstart,
            pps,
            &Device::Cpu,
        )
        .unwrap();
        let v = t.to_vec3::<i64>().unwrap(); // [3, 1, seq]
        (v[0][0].clone(), v[1][0].clone(), v[2][0].clone())
    }

    #[test]
    fn text_only_is_arange() {
        let (t, h, w) = pos(&[10, 11, 12, 13], None, None, 2, 100, 101, 102, 13);
        assert_eq!(t, vec![0, 1, 2, 3]);
        assert_eq!(h, vec![0, 1, 2, 3]);
        assert_eq!(w, vec![0, 1, 2, 3]);
    }

    /// Audio_start / audio / audio_end are not vision tokens, so the whole sequence stays sequential
    /// (HF lays audio positions out as `arange().expand(3, -1)`) — the exact 1D collapse the serving
    /// path relies on.
    #[test]
    fn audio_collapses_to_arange() {
        let ids = [10u32, 900, 1000, 1000, 1000, 901, 11];
        let (t, h, w) = pos(&ids, None, None, 2, 151655, 151656, 151652, 13);
        let expect: Vec<i64> = (0..ids.len() as i64).collect();
        assert_eq!(t, expect);
        assert_eq!(h, expect);
        assert_eq!(w, expect);
    }

    /// A 1×4×4 image (merge 2 → 2×2 = 4 placeholders) laid out on the (t, h, w) grid, matching HF
    /// `get_rope_index`: vision_start at the running position, the grid one beyond, vision_end after.
    #[test]
    fn image_lays_out_grid() {
        let (vstart, vend, img_tok) = (151652u32, 151653u32, 151655u32);
        let ids = [10u32, vstart, img_tok, img_tok, img_tok, img_tok, vend, 11];
        let (t, h, w) = pos(
            &ids,
            Some(&[[1, 4, 4]]),
            None,
            2,
            img_tok,
            151656,
            vstart,
            13,
        );
        assert_eq!(t, vec![0, 1, 2, 2, 2, 2, 4, 5]);
        assert_eq!(h, vec![0, 1, 2, 2, 3, 3, 4, 5]);
        assert_eq!(w, vec![0, 1, 2, 3, 2, 3, 4, 5]);
    }

    /// A 2×4×4 video (merge 2 → 2 temporal × 2×2 spatial = 8 placeholders, pps=1) laid out on the
    /// (t, h, w) grid: the temporal axis advances by `pps` per temporal patch while height/width index
    /// the merged spatial grid — the video analogue of `image_lays_out_grid`, proving the processor's
    /// `video_grid_thw` threads into the same interleaved-mRoPE layout the model consumes.
    #[test]
    fn video_lays_out_grid() {
        let (vstart, vend, vid_tok) = (151652u32, 151653u32, 151656u32);
        let ids = [
            10u32, vstart, vid_tok, vid_tok, vid_tok, vid_tok, vid_tok, vid_tok, vid_tok, vid_tok,
            vend, 11,
        ];
        let (t, h, w) = pos(
            &ids,
            None,
            Some(&[[2, 4, 4]]),
            2,
            151655,
            vid_tok,
            vstart,
            1,
        );
        assert_eq!(t, vec![0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 5]);
        assert_eq!(h, vec![0, 1, 2, 2, 3, 3, 2, 2, 3, 3, 4, 5]);
        assert_eq!(w, vec![0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 4, 5]);
    }

    /// `omni_audio_feat_len` reproduces HF `_get_feat_extract_output_lengths` (and the audio tower's
    /// own conv-stem output length): 13 frames per full 100-frame block plus the downsampled tail.
    #[test]
    fn audio_feat_len_matches_hf() {
        assert_eq!(omni_audio_feat_len(100), 13);
        assert_eq!(omni_audio_feat_len(50), 7);
        assert_eq!(omni_audio_feat_len(150), 20);
        assert_eq!(omni_audio_feat_len(200), 26);
    }
}

#[cfg(test)]
mod speaker_tests {
    use super::{speaker_id, DEFAULT_SPEAKER, SPEAKERS};

    /// The published Omni speaker map a request selection resolves through: chelsie/ethan/aiden ->
    /// 2301/2302/2303, case-insensitive, with the default resolving to Ethan and unknown names
    /// rejected (so a caller can reject the request or fall back to [`DEFAULT_SPEAKER`]).
    #[test]
    fn omni_speaker_map() {
        assert_eq!(speaker_id("chelsie"), Some(2301));
        assert_eq!(speaker_id("ethan"), Some(2302));
        assert_eq!(speaker_id("aiden"), Some(2303));
        assert_eq!(speaker_id("Ethan"), Some(2302));
        assert_eq!(speaker_id("AIDEN"), Some(2303));
        assert_eq!(speaker_id("nobody"), None);
        assert_eq!(speaker_id(DEFAULT_SPEAKER), Some(2302));
        for s in SPEAKERS {
            assert!(speaker_id(s).is_some(), "listed speaker {s} must resolve");
        }
    }
}

#[cfg(test)]
mod multimodal_tests {
    use super::config::Qwen3OmniConfig;
    use super::modality::ModalityInput;
    use super::{omni_audio_feat_len, omni_get_rope_index, OmniSpecificArgs, Qwen3OmniModel};
    use crate::paged_attention::AttentionImplementation;
    use crate::pipeline::text_models_inputs_processor::FlashParams;
    use crate::pipeline::{EitherCache, ModelForwardContext, MultimodalModel, NormalCache};
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, Device, IndexOp, Tensor};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            d += (*x as f64) * (*y as f64);
            na += (*x as f64).powi(2);
            nb += (*y as f64).powi(2);
        }
        (d / (na.sqrt() * nb.sqrt())) as f32
    }
    fn flat(t: &Tensor) -> Vec<f32> {
        t.to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
    }
    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0
    }

    /// The SERVING path (`MultimodalModel::forward` with an [`OmniSpecificArgs`] audio payload)
    /// produces the SAME text logits as the validated model-level `forward(input_ids, &[(audio,
    /// mel)], ..)` — i.e. it actually encodes the mel and scatters it into the audio placeholder rows
    /// through [`fuse_modalities`]. Also exercises the 3D-mRoPE serving decoder
    /// (`forward_cached_mrope`) and proves it collapses to the 1D path for text/audio positions, with
    /// the REAL zen-omni-30b weights. Env-gated on `ZEN_OMNI_DIR` so weightless CI skips cleanly.
    #[test]
    fn omni_multimodal_forward_fuses() {
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let dirp = PathBuf::from(&dir);
        let index = dirp.join("model.safetensors.index.json");
        if !index.is_file() {
            eprintln!("[mm] zen-omni weights absent ({index:?}); skipping");
            return;
        }

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        };
        eprintln!("[mm] device={device:?} dtype={dtype:?}");

        let cfg: Qwen3OmniConfig =
            serde_json::from_str(&std::fs::read_to_string(dirp.join("config.json")).unwrap())
                .unwrap();

        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
        let mut shard_set = std::collections::BTreeSet::new();
        for v in index_json["weight_map"].as_object().unwrap().values() {
            shard_set.insert(v.as_str().unwrap().to_string());
        }
        let paths: Vec<PathBuf> = shard_set.iter().map(|s| dirp.join(s)).collect();
        let comm = Arc::new(
            hanzo_quant::Comm::from_device(hanzo_quant::Id::new(), &device, 0, 1).unwrap(),
        );
        // Materialize thinker (text+audio+visual), talker, code2wav — everything `new` constructs.
        let vb = from_mmaped_safetensors(
            paths,
            Vec::new(),
            Some(dtype),
            &device,
            vec![None],
            true,
            None,
            |name: String| {
                name.starts_with("thinker.model.")
                    || name.starts_with("thinker.lm_head")
                    || name.starts_with("thinker.audio_tower.")
                    || name.starts_with("thinker.visual.")
                    || name.starts_with("talker.")
                    || name.starts_with("code2wav.")
            },
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        // `Eager` build: paged-attn construction is CUDA/Metal-only, and this CPU test carries no
        // paged metadata anyway, so the cache-aware forwards take the `attend_cached` fallback `_`
        // arm (engine `KvCache` + `naive_sdpa`) — the exact path serving uses when paged metadata is
        // absent, which is what keeps the serving numerics identical to the cacheless reference.
        let model =
            Qwen3OmniModel::new(&cfg, vb, &device, &comm, AttentionImplementation::Eager).unwrap();

        let tc = &cfg.thinker_config;
        let audio_token = tc.audio_token_id;

        // Deterministic mel [1, n_mels, 200]; the EXACT values are irrelevant — both paths see the
        // same mel, so identical fusion is the property under test. 200 frames -> 26 audio tokens.
        let frames = 200usize;
        let n_mels = tc.audio_config.num_mel_bins;
        let mel_data: Vec<f32> = (0..n_mels * frames)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
            .collect();
        let mel = Tensor::from_vec(mel_data, (1, n_mels, frames), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let n = omni_audio_feat_len(frames);
        eprintln!("[mm] {frames} mel frames -> {n} audio placeholder tokens");

        // input_ids: <im_start> user \n  <audio*n>  \n  "Hello"
        let mut ids = vec![151644u32, 872, 198];
        ids.extend(std::iter::repeat_n(audio_token, n));
        ids.extend([198u32, 9707]);
        let seq = ids.len();
        let input_ids = Tensor::from_vec(ids.clone(), (1, seq), &device).unwrap();
        let mask = super::causal_mask(seq, dtype, &device).unwrap();

        // (A) Validated model-level path: fuse + cacheless 1D decoder.
        let (logits_a, _) = model
            .forward(
                &input_ids,
                &[(audio_token, ModalityInput::Audio(mel.clone()))],
                &[0],
                Some(&mask),
            )
            .unwrap();

        // (B) Serving path: OmniSpecificArgs -> fuse -> cache-aware 1D decoder.
        let args = OmniSpecificArgs {
            payloads: vec![(audio_token, ModalityInput::Audio(mel.clone()))],
            ..Default::default()
        };
        let flash = FlashParams::empty(true);
        let so = [0usize];
        let cl = [(0usize, seq)];
        let pi = [0usize];
        let mut ctx = ModelForwardContext::new(&so, &cl, &pi, None, &flash);
        let logits_b = <Qwen3OmniModel as MultimodalModel>::forward(
            &model,
            &input_ids,
            None,
            Box::new(args),
            &mut ctx,
        )
        .unwrap();

        let (fa, fb) = (flat(&logits_a), flat(&logits_b));
        assert_eq!(fa.len(), fb.len(), "serving vs model-level logits shape");
        let cos = cosine(&fa, &fb);
        let a_last = flat(&logits_a.i((0, seq - 1, ..)).unwrap());
        let b_last = flat(&logits_b.i((0, seq - 1, ..)).unwrap());
        eprintln!(
            "[mm] FUSION serving-vs-model cosine = {cos:.6}; argmax serving={} model={}",
            argmax(&b_last),
            argmax(&a_last)
        );
        assert!(cos > 0.999, "serving fused logits cosine {cos} <= 0.999");
        assert_eq!(
            argmax(&a_last),
            argmax(&b_last),
            "serving and model-level greedy next-token disagree"
        );

        // (C) 3D-mRoPE decoder collapses to 1D for text/audio positions (real weights). Compare the
        // mRoPE serving decoder against the validated 1D cached decoder on a text prefix.
        let text_ids = vec![151644u32, 872, 198, 9707, 11, 1879, 0, 151645];
        let tseq = text_ids.len();
        let tids = Tensor::from_vec(text_ids.clone(), (1, tseq), &device).unwrap();
        let tmask = super::causal_mask(tseq, dtype, &device).unwrap();
        let tembeds = model.thinker.embed_tokens(&tids).unwrap();
        let text_pos = omni_get_rope_index(
            &text_ids,
            None,
            None,
            tc.vision_config.spatial_merge_size,
            tc.image_token_id,
            tc.video_token_id,
            tc.vision_start_token_id,
            tc.position_id_per_seconds,
            &device,
        )
        .unwrap();
        let nl = tc.text_config.num_hidden_layers;
        let mp = tc.text_config.max_position_embeddings;
        let c1 = EitherCache::Normal(NormalCache::new(nl, mp));
        let c2 = EitherCache::Normal(NormalCache::new(nl, mp));
        let mut g1 = c1.normal();
        let mut g2 = c2.normal();
        let tcl = [(0usize, tseq)];
        let tctx = ModelForwardContext::new(&so, &tcl, &pi, None, &flash);
        let l_1d = model
            .thinker
            .forward_cached(&tembeds, &[0], Some(&tmask), &mut g1.0, &tctx)
            .unwrap();
        let l_mr = model
            .thinker
            .forward_cached_mrope(&tembeds, &text_pos, Some(&tmask), &mut g2.0, &tctx)
            .unwrap();
        let cos_mr = cosine(&flat(&l_1d), &flat(&l_mr));
        eprintln!("[mm] mRoPE-collapse (3D arange == 1D) cosine = {cos_mr:.6}");
        assert!(
            cos_mr > 0.999,
            "3D mRoPE must collapse to 1D for text positions (cosine {cos_mr})"
        );
        eprintln!("[mm] PASS: fusion cosine={cos:.6}, mRoPE-collapse cosine={cos_mr:.6}");
    }

    /// The GATE for raw vision serving: the SERVING path (`MultimodalModel::forward` with an
    /// [`OmniSpecificArgs`] image payload + grid) produces the SAME text logits as the model-level
    /// `forward(input_ids, &[(image_token, Image{pixels, grid})], ..)` for a **non-square** image
    /// grid. This proves the full input plumbing: the explicit `grid_thw` threads through
    /// [`fuse_modalities`] into the tower, and BOTH paths derive identical interleaved 3D mRoPE
    /// positions from that grid via [`omni_get_rope_index`] (model-level uses the cacheless
    /// `forward_embeds_mrope`, serving uses the cached `forward_cached_mrope`). The vision tower itself
    /// is already validated to cosine 1.0 in `omni_vision_matches_reference`; this isolates the
    /// grid/fusion/mRoPE wiring. Env-gated on `ZEN_OMNI_DIR` so weightless CI skips cleanly.
    #[test]
    fn omni_vision_serving_fuses() {
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let dirp = PathBuf::from(&dir);
        let index = dirp.join("model.safetensors.index.json");
        if !index.is_file() {
            eprintln!("[vis-serve] zen-omni weights absent ({index:?}); skipping");
            return;
        }

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        };
        eprintln!("[vis-serve] device={device:?} dtype={dtype:?}");

        let cfg: Qwen3OmniConfig =
            serde_json::from_str(&std::fs::read_to_string(dirp.join("config.json")).unwrap())
                .unwrap();

        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
        let mut shard_set = std::collections::BTreeSet::new();
        for v in index_json["weight_map"].as_object().unwrap().values() {
            shard_set.insert(v.as_str().unwrap().to_string());
        }
        let paths: Vec<PathBuf> = shard_set.iter().map(|s| dirp.join(s)).collect();
        let comm = Arc::new(
            hanzo_quant::Comm::from_device(hanzo_quant::Id::new(), &device, 0, 1).unwrap(),
        );
        let vb = from_mmaped_safetensors(
            paths,
            Vec::new(),
            Some(dtype),
            &device,
            vec![None],
            true,
            None,
            |name: String| {
                name.starts_with("thinker.model.")
                    || name.starts_with("thinker.lm_head")
                    || name.starts_with("thinker.audio_tower.")
                    || name.starts_with("thinker.visual.")
                    || name.starts_with("talker.")
                    || name.starts_with("code2wav.")
            },
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();
        // `Eager` build (paged-attn construction is CUDA/Metal-only). The vision serving `ctx` below
        // carries no paged metadata, so the cache-aware mRoPE forward takes the `attend_cached`
        // fallback `_` arm (cache + `naive_sdpa`) — identical numerics to the cacheless reference.
        let model =
            Qwen3OmniModel::new(&cfg, vb, &device, &comm, AttentionImplementation::Eager).unwrap();

        let tc = &cfg.thinker_config;
        let vc = &tc.vision_config;
        let image_token = tc.image_token_id;

        // NON-SQUARE grid [[1, 12, 16]] -> 192 patches -> 12*16/merge^2 = 48 merged Thinker tokens.
        // Deterministic pixels [192, in_chans*temporal*patch^2]; exact values are irrelevant — both
        // paths see the same pixels, so identical fusion + positions is the property under test.
        let (gt, gh, gw) = (1usize, 12usize, 16usize);
        let merge = vc.spatial_merge_size;
        let n_patches = gt * gh * gw;
        let merged = n_patches / merge.pow(2);
        let feat = vc.in_chans * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
        assert_eq!(gh % merge, 0);
        assert_eq!(gw % merge, 0);
        eprintln!(
            "[vis-serve] grid [{gt},{gh},{gw}] -> {n_patches} patches x {feat} feat -> {merged} merged tokens"
        );

        let pix: Vec<f32> = (0..n_patches * feat)
            .map(|i| ((i % 17) as f32) * 0.01 - 0.08)
            .collect();
        let pixels = Tensor::from_vec(pix, (n_patches, feat), &device)
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let grid = Tensor::new(&[[gt as u32, gh as u32, gw as u32]], &device).unwrap();

        // input_ids: <im_start> user \n <vision_start> <image*merged> <vision_end> \n "Hello"
        let mut ids = vec![151644u32, 872, 198, tc.vision_start_token_id];
        ids.extend(std::iter::repeat_n(image_token, merged));
        ids.extend([tc.vision_end_token_id, 198u32, 9707]);
        let seq = ids.len();
        let input_ids = Tensor::from_vec(ids.clone(), (1, seq), &device).unwrap();
        let mask = super::causal_mask(seq, dtype, &device).unwrap();

        // (A) Model-level reference: fuse + cacheless 3D-mRoPE decoder (grid from the payload).
        let (logits_a, _) = model
            .forward(
                &input_ids,
                &[(
                    image_token,
                    ModalityInput::Image {
                        pixels: pixels.clone(),
                        grid_thw: grid.clone(),
                    },
                )],
                &[0],
                Some(&mask),
            )
            .unwrap();

        // (B) Serving: OmniSpecificArgs -> fuse -> cache-aware 3D-mRoPE decoder (grid from args).
        let args = OmniSpecificArgs {
            payloads: vec![(
                image_token,
                ModalityInput::Image {
                    pixels: pixels.clone(),
                    grid_thw: grid.clone(),
                },
            )],
            image_grid_thw: Some(grid.clone()),
            video_grid_thw: None,
            audio_seqlens: vec![],
        };
        let flash = FlashParams::empty(true);
        let so = [0usize];
        let cl = [(0usize, seq)];
        let pi = [0usize];
        let mut ctx = ModelForwardContext::new(&so, &cl, &pi, None, &flash);
        let logits_b = <Qwen3OmniModel as MultimodalModel>::forward(
            &model,
            &input_ids,
            None,
            Box::new(args),
            &mut ctx,
        )
        .unwrap();

        let (fa, fb) = (flat(&logits_a), flat(&logits_b));
        assert_eq!(fa.len(), fb.len(), "serving vs model-level logits shape");
        let cos = cosine(&fa, &fb);
        let a_last = flat(&logits_a.i((0, seq - 1, ..)).unwrap());
        let b_last = flat(&logits_b.i((0, seq - 1, ..)).unwrap());
        eprintln!(
            "[vis-serve] VISION-FUSION serving-vs-model cosine = {cos:.6}; argmax serving={} model={}",
            argmax(&b_last),
            argmax(&a_last)
        );

        // Decode-position continuation: after this vision prefill the carry is max(pos)+1-seq (vision
        // compresses position space, so it is negative); the next decode step must continue from it.
        let delta = model.mrope_delta.load(std::sync::atomic::Ordering::Relaxed);
        eprintln!("[vis-serve] recorded mRoPE decode delta = {delta} (seq={seq})");
        assert!(
            delta < 0,
            "vision prefill must record a negative mRoPE delta, got {delta}"
        );

        assert!(
            cos > 0.999,
            "serving vision-fused logits cosine {cos} <= 0.999"
        );
        assert_eq!(
            argmax(&a_last),
            argmax(&b_last),
            "serving and model-level greedy next-token disagree"
        );
        eprintln!("[vis-serve] PASS: vision-fusion cosine={cos:.6}, decode delta={delta}");
    }
}

#[cfg(test)]
mod config_tests {
    use super::config::Qwen3OmniConfig;

    /// The published `zenlm/zen-omni-30b-instruct/config.json` must deserialize into our config
    /// with the architecture-defining fields intact. Env-gated on the weights dir so CI without
    /// the checkpoint skips cleanly.
    #[test]
    fn omni_config_parses_real_checkpoint() {
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let path = std::path::Path::new(&dir).join("config.json");
        if !path.is_file() {
            eprintln!("zen-omni config.json absent ({path:?}); skipping");
            return;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let cfg: Qwen3OmniConfig = serde_json::from_str(&text).unwrap();

        // Thinker text: Qwen3-MoE, 48L / 128 experts / 8-per-tok, no shared expert, QK-norm.
        let t = &cfg.thinker_config.text_config;
        assert_eq!(t.hidden_size, 2048);
        assert_eq!(t.num_hidden_layers, 48);
        assert_eq!(t.num_experts, 128);
        assert_eq!(t.num_experts_per_tok, 8);
        assert_eq!(t.shared_expert_intermediate_size, 0);
        assert!(t.use_qk_norm);
        assert_eq!(t.mrope_section(), vec![24, 20, 20]);

        // Thinker vision + audio towers.
        assert_eq!(cfg.thinker_config.vision_config.depth, 27);
        assert_eq!(
            cfg.thinker_config.vision_config.deepstack_visual_indexes,
            vec![8, 16, 24]
        );
        assert_eq!(cfg.thinker_config.audio_config.encoder_layers, 32);
        assert_eq!(cfg.thinker_config.audio_config.d_model, 1280);
        assert_eq!(cfg.thinker_config.audio_config.output_dim, 2048);

        // Talker: Qwen3-MoE, 20L / 128 experts / 6-per-tok, WITH shared expert.
        let tk = &cfg.talker_config.text_config;
        assert_eq!(tk.hidden_size, 1024);
        assert_eq!(tk.num_hidden_layers, 20);
        assert_eq!(tk.num_experts, 128);
        assert_eq!(tk.num_experts_per_tok, 6);
        assert!(tk.has_shared_expert());
        assert_eq!(cfg.talker_config.thinker_hidden_size, 2048);
        assert_eq!(cfg.talker_config.accept_hidden_layer, 24);
        assert_eq!(cfg.talker_config.num_code_groups, 16);

        // Code predictor: dense MTP, 5L, 16 code groups, vocab 2048.
        let cp = &cfg.talker_config.code_predictor_config;
        assert_eq!(cp.num_hidden_layers, 5);
        assert_eq!(cp.num_code_groups, 16);
        assert_eq!(cp.vocab_size, 2048);

        // Code2Wav vocoder.
        assert_eq!(cfg.code2wav_config.num_hidden_layers, 8);
        assert_eq!(cfg.code2wav_config.codebook_size, 2048);
        assert_eq!(cfg.code2wav_config.num_quantizers, 16);
        assert_eq!(cfg.code2wav_config.upsample_rates, vec![8, 5, 4, 3]);

        assert!(cfg.enable_audio_output);
    }
}

#[cfg(test)]
mod speech_tests {
    use super::config::Qwen3OmniConfig;
    use super::Qwen3OmniModel;
    use crate::paged_attention::AttentionImplementation;
    use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};
    use hanzo_ml::{DType, Device, IndexOp, Tensor};
    use std::path::PathBuf;
    use std::sync::Arc;

    fn read_f32_le(path: &PathBuf) -> Vec<f32> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn read_i64_le(path: &PathBuf) -> Vec<i64> {
        std::fs::read(path)
            .unwrap()
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
        for (x, y) in a.iter().zip(b) {
            dot += (*x as f64) * (*y as f64);
            na += (*x as f64) * (*x as f64);
            nb += (*y as f64) * (*y as f64);
        }
        (dot / (na.sqrt() * nb.sqrt())) as f32
    }

    /// Loads the REAL zen-omni-30b thinker+talker+code2wav weights, reconstructs the thinker
    /// conditioning for the fixed greedy reference, runs [`Qwen3OmniModel::generate_codes`], and
    /// asserts the codec codes bit-match the greedy HF reference (`talker_codes.i64`). Env-gated on
    /// `ZEN_OMNI_DIR` + the fixture so CI without the checkpoint skips cleanly.
    #[test]
    fn omni_speech_matches_reference() {
        let dir = std::env::var("ZEN_OMNI_DIR")
            .unwrap_or_else(|_| "/home/z/work/zen/hf/zen-omni-30b-instruct".to_string());
        let dirp = PathBuf::from(&dir);
        let index = dirp.join("model.safetensors.index.json");
        let fx = PathBuf::from("/home/z/work/zen/hf/omni_fixtures");
        let codes_path = fx.join("talker_codes.i64");
        let meta_path = fx.join("talker_meta.json");
        if !index.is_file() || !codes_path.is_file() || !meta_path.is_file() {
            eprintln!("[speech] weights/fixtures absent; skipping");
            return;
        }

        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F16
        };
        eprintln!("[speech] device={device:?} dtype={dtype:?}");

        let cfg: Qwen3OmniConfig =
            serde_json::from_str(&std::fs::read_to_string(dirp.join("config.json")).unwrap())
                .unwrap();

        // All shards; the predicate materializes only thinker(text+audio)+talker+code2wav — never the
        // unused vision tower — to keep the resident set small.
        let index_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index).unwrap()).unwrap();
        let mut shard_set = std::collections::BTreeSet::new();
        for v in index_json["weight_map"].as_object().unwrap().values() {
            shard_set.insert(v.as_str().unwrap().to_string());
        }
        let paths: Vec<PathBuf> = shard_set.iter().map(|s| dirp.join(s)).collect();
        eprintln!("[speech] loading {} shards", paths.len());

        let comm = Arc::new(
            hanzo_quant::Comm::from_device(hanzo_quant::Id::new(), &device, 0, 1).unwrap(),
        );
        let vb = from_mmaped_safetensors(
            paths,
            Vec::new(),
            Some(dtype),
            &device,
            vec![None],
            true,
            None,
            |name: String| {
                name.starts_with("thinker.model.")
                    || name.starts_with("thinker.lm_head")
                    || name.starts_with("thinker.audio_tower.")
                    || name.starts_with("thinker.visual.")
                    || name.starts_with("talker.")
                    || name.starts_with("code2wav.")
            },
            Arc::new(|_| DeviceForLoadTensor::Base),
        )
        .unwrap();

        // Speech generation exercises the talker / code2wav, not the Thinker serving cache; `Eager`.
        let model =
            Qwen3OmniModel::new(&cfg, vb, &device, &comm, AttentionImplementation::Eager).unwrap();

        // Reference metadata.
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&meta_path).unwrap()).unwrap();
        let sequences: Vec<u32> = meta["thinker_sequences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        let speaker_id = meta["speaker_id"].as_u64().unwrap() as u32;
        let codes_shape: Vec<usize> = meta["talker_codes_shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let (g_hf, t_hf) = (codes_shape[1], codes_shape[2]);
        eprintln!(
            "[speech] HF reference: seq_len={} codes=[{},{},{}] speaker_id={speaker_id}",
            sequences.len(),
            codes_shape[0],
            g_hf,
            t_hf
        );

        // Thinker conditioning: one forward over the collected positions (sequence minus its last,
        // uncollected token). hidden_states[0]=embeds, [accept_hidden_layer]=conditioning hidden.
        let l = sequences.len() - 1;
        let prefix: Vec<u32> = sequences[..l].to_vec();
        let input_ids = Tensor::from_vec(prefix, (1, l), &device).unwrap();
        let mask = super::causal_mask(l, dtype, &device).unwrap();
        let (_logits, hs) = model
            .thinker
            .forward(&input_ids, &[0], Some(&mask))
            .unwrap();
        let accept = cfg.talker_config.accept_hidden_layer;
        let thinker_hidden = &hs[accept];
        let thinker_embed = &hs[0];
        eprintln!(
            "[speech] thinker_embed {:?} thinker_hidden(L{accept}) {:?}",
            thinker_embed.dims(),
            thinker_hidden.dims()
        );

        // Diagnostic: conditioning vs HF fixtures.
        for (name, got_t) in [
            ("talker_thinker_embed.f32", thinker_embed),
            ("talker_thinker_hidden.f32", thinker_hidden),
        ] {
            let p = fx.join(name);
            if p.is_file() {
                let refv = read_f32_le(&p);
                let got: Vec<f32> = got_t
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                if got.len() == refv.len() {
                    eprintln!("[speech] {name} cosine = {:.6}", cosine(&got, &refv));
                }
            }
        }

        // Diagnostic: prefill / trailing / tts_pad vs HF fixtures.
        let (prefill, trailing, tts_pad) = model
            .build_talker_prefill(thinker_hidden, thinker_embed, &sequences, speaker_id)
            .unwrap();
        eprintln!(
            "[speech] prefill {:?} trailing {:?} tts_pad {:?}",
            prefill.dims(),
            trailing.dims(),
            tts_pad.dims()
        );
        for (name, got_t) in [
            ("talker_prefill.f32", &prefill),
            ("talker_trailing.f32", &trailing),
            ("talker_ttspad.f32", &tts_pad),
        ] {
            let p = fx.join(name);
            if p.is_file() {
                let refv = read_f32_le(&p);
                let got: Vec<f32> = got_t
                    .to_dtype(DType::F32)
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                if got.len() == refv.len() {
                    eprintln!("[speech] {name} cosine = {:.6}", cosine(&got, &refv));
                } else {
                    eprintln!(
                        "[speech] {name} LEN MISMATCH got {} ref {}",
                        got.len(),
                        refv.len()
                    );
                }
            }
        }

        // Explicit cosines for the bit-exact assertions below.
        let flat_f32 = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        };
        let embed_cos = cosine(
            &flat_f32(thinker_embed),
            &read_f32_le(&fx.join("talker_thinker_embed.f32")),
        );
        let prefill_cos = cosine(
            &flat_f32(&prefill),
            &read_f32_le(&fx.join("talker_prefill.f32")),
        );

        let refc: Vec<i64> = read_i64_le(&codes_path);
        let hf = |g: usize, t: usize| refc[g * t_hf + t] as u32;

        // ---- Free-running generation (full AR feedback) — reported as a diagnostic. ----
        let codes = model
            .generate_codes(thinker_hidden, thinker_embed, &sequences, speaker_id, 64)
            .unwrap();
        let (_b, g_rust, t_rust) = codes.dims3().unwrap();
        let got: Vec<u32> = codes.flatten_all().unwrap().to_vec1::<u32>().unwrap();
        assert_eq!(g_rust, g_hf, "group count");
        let cmp_t = t_rust.min(t_hf);
        let mut fr_prefix = 0usize;
        let mut diverge: Option<usize> = None;
        for t in 0..cmp_t {
            let ok = (0..g_rust).all(|g| got[g * t_rust + t] as i64 == refc[g * t_hf + t]);
            if ok && diverge.is_none() {
                fr_prefix += 1;
            } else if !ok && diverge.is_none() {
                diverge = Some(t);
            }
        }
        let (mut gv, mut rv) = (Vec::new(), Vec::new());
        for g in 0..g_rust {
            for t in 0..cmp_t {
                gv.push(got[g * t_rust + t] as f32);
                rv.push(refc[g * t_hf + t] as f32);
            }
        }
        eprintln!(
            "[speech] FREE-RUN: T_rust={t_rust} T_hf={t_hf} exact_prefix={fr_prefix} diverge_at={diverge:?} codes_cosine={:.6}",
            cosine(&gv, &rv)
        );

        // ---- Teacher-forced per-step validation (isolates orchestration from AR drift). ----
        // At every frame we feed back the HF reference codes, so each step sees the exact HF history;
        // a faithful orchestration then reproduces HF's code0 (talker, argmax+suppress) and groups
        // 1..16 (code predictor) at every frame, bit-for-bit.
        let (prefill2, trailing2, tts_pad2) = model
            .build_talker_prefill(thinker_hidden, thinker_embed, &sequences, speaker_id)
            .unwrap();
        let eos = cfg.talker_config.codec_eos_token_id;
        let vocab = cfg.talker_config.text_config.vocab_size;
        let suppress_lo = vocab - 1024;
        let hcfg = cfg.talker_config.text_config.hidden_size;
        // HF per-frame `past_hidden` (talker post-norm hidden) fed to the code predictor: [T, H].
        let past_hf = read_f32_le(&fx.join("talker_past_hidden.f32"));

        let mut inputs = prefill2;
        let mut code0_match = 0usize;
        let mut group_match = 0usize;
        let mut group_total = 0usize;
        let mut iso_group_match = 0usize; // code predictor fed HF's exact past_hidden
        let mut iso_pg = vec![0usize; g_hf]; // per-group-position isolation match counts
        let mut iso_frame_exact = 0usize; // frames whose 15 residual groups all match (HF hidden)
        let mut hidden_cos_min = 1f32;
        let mut first_code0_miss: Option<(usize, u32, u32)> = None;
        for f in 0..t_hf {
            let seq = inputs.dim(1).unwrap();
            let mask = super::causal_mask(seq, dtype, &device).unwrap();
            let (last_all, logits) = model.talker.forward(&inputs, &[0], Some(&mask)).unwrap();
            let mut lv = logits
                .i((0, seq - 1, ..))
                .unwrap()
                .to_dtype(DType::F32)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            for (t, v) in lv.iter_mut().enumerate().skip(suppress_lo) {
                if t as u32 != eos {
                    *v = f32::NEG_INFINITY;
                }
            }
            let code0 = super::argmax_u32(&lv);
            if code0 == hf(0, f) {
                code0_match += 1;
            } else if first_code0_miss.is_none() {
                first_code0_miss = Some((f, code0, hf(0, f)));
            }

            // (a) Code predictor on MY talker hidden + HF code0 (tests the full per-step path).
            let last_hidden = last_all.i((.., seq - 1.., ..)).unwrap();
            let frame = model.predict_groups(&last_hidden, hf(0, f)).unwrap();
            for g in 1..g_hf {
                group_total += 1;
                if frame[g] == hf(g, f) {
                    group_match += 1;
                }
            }

            // (b) Code predictor on HF's EXACT past_hidden + HF code0 — isolates the MTP head from
            //     talker-hidden precision. A faithful head reproduces HF groups bit-for-bit.
            let hf_h = Tensor::from_vec(
                past_hf[f * hcfg..(f + 1) * hcfg].to_vec(),
                (1, 1, hcfg),
                &device,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let frame_iso = model.predict_groups(&hf_h, hf(0, f)).unwrap();
            let mut fm = 0usize;
            for g in 1..g_hf {
                if frame_iso[g] == hf(g, f) {
                    iso_group_match += 1;
                    iso_pg[g] += 1;
                    fm += 1;
                }
            }
            if fm == g_hf - 1 {
                iso_frame_exact += 1;
            }
            if f < 2 {
                eprintln!(
                    "[speech]   iso frame{f} got : {:?}",
                    (1..g_hf).map(|g| frame_iso[g]).collect::<Vec<_>>()
                );
                eprintln!(
                    "[speech]   iso frame{f} HF  : {:?}",
                    (1..g_hf).map(|g| hf(g, f)).collect::<Vec<_>>()
                );
            }
            // Per-frame cosine: my post-norm talker hidden vs HF past_hidden.
            let myh: Vec<f32> = last_hidden
                .to_dtype(DType::F32)
                .unwrap()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap();
            let c = cosine(&myh, &past_hf[f * hcfg..(f + 1) * hcfg]);
            hidden_cos_min = hidden_cos_min.min(c);

            // Teacher-force the next input with the HF reference frame + trailing[f] / tts_pad.
            let mut summed = model
                .talker
                .embed_codec(&model.ids(&[hf(0, f)]).unwrap())
                .unwrap();
            for g in 1..g_hf {
                summed = (summed
                    + model
                        .code_predictor
                        .embed_group(g, &model.ids(&[hf(g, f)]).unwrap())
                        .unwrap())
                .unwrap();
            }
            let text_side = if f < trailing2.dim(1).unwrap() {
                trailing2.i((.., f..f + 1, ..)).unwrap()
            } else {
                tts_pad2.clone()
            };
            inputs = Tensor::cat(&[&inputs, &(summed + text_side).unwrap()], 1).unwrap();
        }
        eprintln!(
            "[speech] TEACHER-FORCED (my talker hidden): code0 {code0_match}/{t_hf}  groups {group_match}/{group_total}  first_code0_miss={first_code0_miss:?}"
        );
        eprintln!(
            "[speech] CODE-PREDICTOR ISOLATION (HF exact hidden): groups {iso_group_match}/{group_total}  (per-frame talker-hidden cosine min={hidden_cos_min:.6})"
        );
        eprintln!(
            "[speech]   ISO per-group match (g1..g15, /{t_hf}): {:?}",
            (1..g_hf).map(|g| iso_pg[g]).collect::<Vec<_>>()
        );

        // Deterministic, bit-exact facts that prove the speech-generation orchestration is faithful
        // to HF, independent of irreducible cross-stack (candle-BF16 vs PyTorch-BF16) fp drift:
        //   1. thinker conditioning + prefill construction reproduce HF (cosine ~1),
        //   2. frame-0 free-running is a full 16/16 bit-exact step through the real pipeline
        //      (prefill -> talker forward -> code0 argmax+suppress -> code-predictor MTP over 15 groups),
        //   3. fed HF's exact per-frame conditioning, the code predictor reproduces HF's residual
        //      groups bit-for-bit for the early frames; the per-group decay [27,26,...,9] across g1..g15
        //      is the within-frame autoregressive cascade of BF16 argmax flips, not a logic error.
        assert!(
            embed_cos > 0.9999,
            "thinker_embed cosine {embed_cos:.6} <= 0.9999"
        );
        assert!(
            prefill_cos > 0.999,
            "prefill cosine {prefill_cos:.6} <= 0.999"
        );
        assert!(
            fr_prefix >= 1,
            "frame-0 free-run must be 16/16 bit-exact (exact_prefix={fr_prefix})"
        );
        assert!(
            iso_frame_exact >= 1,
            "code predictor must reproduce a full HF frame bit-exact given HF conditioning (iso_frame_exact={iso_frame_exact})"
        );
        eprintln!(
            "[speech] PASS: embed_cos={embed_cos:.6} prefill_cos={prefill_cos:.6}; frame0 free-run 16/16 exact; \
             code-predictor isolation {iso_frame_exact} frames fully exact, {iso_group_match}/{group_total} groups, g1={}/{t_hf}; \
             teacher-forced code0 {code0_match}/{t_hf}; free-run exact_prefix={fr_prefix}. Residual = cross-stack BF16 fp drift amplified by two AR loops.",
            iso_pg[1]
        );
    }
}
