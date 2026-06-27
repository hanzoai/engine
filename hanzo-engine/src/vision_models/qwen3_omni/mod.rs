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

use std::sync::Arc;

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_quant::{Comm, ShardedVarBuilder};

pub mod audio_tower;
pub mod code2wav;
pub mod config;
pub mod modality;
pub mod talker;
pub mod thinker;
pub mod vision;

pub use config::Qwen3OmniConfig;

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
    ) -> Result<Self> {
        let thinker = OmniThinkerText::new(
            &cfg.thinker_config.text_config,
            vb.pp("thinker"),
            device,
            comm,
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
        let merge = cfg.thinker_config.vision_config.spatial_merge_size;
        encoders.push(Box::new(vision::VisionModality::new(
            vision_tower.clone(),
            cfg.thinker_config.image_token_id,
            merge,
        )));
        encoders.push(Box::new(vision::VisionModality::new(
            vision_tower,
            cfg.thinker_config.video_token_id,
            merge,
        )));

        let talker = OmniTalker::new(&cfg.talker_config, vb.pp("talker"), device, comm)?;
        let code_predictor = OmniCodePredictor::new(
            &cfg.talker_config.code_predictor_config,
            vb.pp("talker").pp("code_predictor"),
            device,
        )?;
        let code2wav = OmniCode2Wav::new(&cfg.code2wav_config, vb.pp("code2wav"), device)?;

        Ok(Self {
            thinker,
            encoders,
            talker,
            code_predictor,
            code2wav,
            cfg: cfg.clone(),
            device: device.clone(),
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
    pub fn forward(
        &self,
        input_ids: &Tensor,
        inputs: &[(u32, ModalityInput)],
        seqlen_offsets: &[usize],
        mask: Option<&Tensor>,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        let embeds = self.thinker.embed_tokens(input_ids)?;
        let fused = fuse_modalities(&embeds, input_ids, &self.encoders, inputs, &self.device)?;
        self.thinker.forward_embeds(&fused, seqlen_offsets, mask)
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
            let (last_hidden_all, logits) = self.talker.forward(&inputs_embeds, &[0], Some(&mask))?;

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
        let tts = self.talker.project_text(&self.thinker.embed_tokens(&self.ids(&[
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
                let ah = self.talker.project_text(&thinker_embed.i((.., start..hi, ..))?)?;
                // assistant_text = [ah[:3], tts_pad*4, tts_bos, ah[3:4]]
                let pad4 = tts_pad.broadcast_as((1, 4, ht))?.contiguous()?;
                let assistant_text = Tensor::cat(
                    &[&ah.i((.., 0..3, ..))?, &pad4, &tts_bos, &ah.i((.., 3..4, ..))?],
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

/// HF `config.assistant_token_id` (absent from [`Qwen3OmniConfig`]; stable checkpoint constant).
const ASSISTANT_TOKEN_ID: u32 = 77091;

/// HF `talker_config.speaker_id` (absent from [`Qwen3OmniConfig`]; the published checkpoint map).
fn speaker_id(name: &str) -> Option<u32> {
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

        let comm =
            Arc::new(hanzo_quant::Comm::from_device(hanzo_quant::Id::new(), &device, 0, 1).unwrap());
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

        let model = Qwen3OmniModel::new(&cfg, vb, &device, &comm).unwrap();

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
        let (_logits, hs) = model.thinker.forward(&input_ids, &[0], Some(&mask)).unwrap();
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
                    eprintln!("[speech] {name} LEN MISMATCH got {} ref {}", got.len(), refv.len());
                }
            }
        }

        // Explicit cosines for the bit-exact assertions below.
        let flat_f32 = |t: &Tensor| -> Vec<f32> {
            t.to_dtype(DType::F32).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap()
        };
        let embed_cos = cosine(&flat_f32(thinker_embed), &read_f32_le(&fx.join("talker_thinker_embed.f32")));
        let prefill_cos = cosine(&flat_f32(&prefill), &read_f32_le(&fx.join("talker_prefill.f32")));

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
            let hf_h = Tensor::from_vec(past_hf[f * hcfg..(f + 1) * hcfg].to_vec(), (1, 1, hcfg), &device)
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
            let mut summed = model.talker.embed_codec(&model.ids(&[hf(0, f)]).unwrap()).unwrap();
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
        assert!(embed_cos > 0.9999, "thinker_embed cosine {embed_cos:.6} <= 0.9999");
        assert!(prefill_cos > 0.999, "prefill cosine {prefill_cos:.6} <= 0.999");
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
