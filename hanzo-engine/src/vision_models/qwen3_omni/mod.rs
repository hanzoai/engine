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

        // Audio is the one native modality wired now; vision/video register the same way later.
        let audio_tower = OmniAudioTower::new(
            &cfg.thinker_config.audio_config,
            vb.pp("thinker").pp("audio_tower"),
            device,
        )?;
        let encoders: Vec<Box<dyn ModalityEncoder>> = vec![Box::new(AudioModality(
            audio_tower,
            cfg.thinker_config.audio_token_id,
        ))];

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

    /// Speak: render a Thinker hidden state (layer `accept_hidden_layer`, `[1, T, thinker_hidden]`)
    /// into a 24 kHz waveform `[1, 1, samples]`. Greedy (argmax).
    ///
    /// Structure mirrors the qwen3_tts `generate_codes` loop with the Omni components: build the
    /// talker prefill, then per frame run `talker.forward` -> argmax codebook-0 -> the code-predictor
    /// MTP over groups `1..num_code_groups` -> feed the summed frame embeddings back; finally assemble
    /// `[1, 16, T]` codes and `code2wav.decode`.
    pub fn generate_speech(&self, thinker_hidden: &Tensor, max_frames: usize) -> Result<Tensor> {
        let tc = &self.cfg.talker_config;
        let groups = tc.num_code_groups;
        let eos = tc.codec_eos_token_id;
        let dtype = thinker_hidden.dtype();
        let t = thinker_hidden.dim(1)?;

        // Prefill (qwen3_tts-analogous; see report note): the projected Thinker hidden is the
        // per-position text-side conditioning, the codec side is `codec_pad` over those positions,
        // then one trailing step of (projected pad-text + codec BOS) from which frame 0 is produced.
        let cond = self.talker.project_thinker_hidden(thinker_hidden)?; // [1, T, Ht]
        let codec_pad = self.talker.embed_codec(&self.ids(&vec![tc.codec_pad_id; t])?)?; // [1, T, Ht]
        let body = (cond + codec_pad)?;

        // Trailing text-side hidden fed alongside every generated codec frame: the projected Thinker
        // embedding of the TTS pad token (qwen3_tts feeds `tts_pad` as the trailing text hidden).
        let pad_emb = self
            .thinker
            .embed_tokens(&self.ids(&[self.cfg.tts_pad_token_id])?)?;
        let trailing = self.talker.project_text(&pad_emb)?; // [1, 1, Ht]
        let codec_bos = self.talker.embed_codec(&self.ids(&[tc.codec_bos_id])?)?; // [1, 1, Ht]
        let bos_step = trailing.broadcast_add(&codec_bos)?;

        let mut inputs_embeds = Tensor::cat(&[&body, &bos_step], 1)?;

        let mut all_codes: Vec<Vec<u32>> = Vec::new();
        for _ in 0..max_frames {
            let mask = causal_mask(inputs_embeds.dim(1)?, dtype, &self.device)?;
            let (hidden, logits) = self.talker.forward(&inputs_embeds, &[0], Some(&mask))?;
            let last_hidden = hidden.i((.., hidden.dim(1)? - 1.., ..))?;
            let code0 = logits
                .i((0, logits.dim(1)? - 1, ..))?
                .argmax(D::Minus1)?
                .to_scalar::<u32>()?;
            if code0 == eos {
                break;
            }
            let frame = self.predict_groups(&last_hidden, code0)?;

            // Next talker input = sum of all group embeddings + trailing text-side hidden.
            let mut summed = self.talker.embed_codec(&self.ids(&[code0])?)?;
            for (group, &id) in frame.iter().enumerate().skip(1) {
                summed = (summed + self.code_predictor.embed_group(group, &self.ids(&[id])?)?)?;
            }
            let next = summed.broadcast_add(&trailing)?;
            all_codes.push(frame);
            inputs_embeds = Tensor::cat(&[&inputs_embeds, &next], 1)?;
        }

        if all_codes.is_empty() {
            return Tensor::zeros((1usize, 1usize, 0usize), DType::F32, &self.device);
        }

        // Assemble group-major `[1, groups, T]` and decode to a waveform.
        let frames = all_codes.len();
        let mut flat = Vec::with_capacity(groups * frames);
        for g in 0..groups {
            for frame in &all_codes {
                flat.push(frame[g]);
            }
        }
        let codes = Tensor::from_vec(flat, (1, groups, frames), &self.device)?;
        self.code2wav.decode(&codes)
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
