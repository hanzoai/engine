use serde::{Deserialize, Serialize};

// Qwen3-TTS (zen3-tts) config. Mirrors the HF `Qwen3TTSForConditionalGeneration` config.json
// plus the separate `speech_tokenizer/config.json` (the WavTokenizer-style codec / vocoder).

fn default_position_id_per_seconds() -> usize {
    13
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_code_groups: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_code_groups: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub text_hidden_size: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_position_id_per_seconds")]
    pub position_id_per_seconds: usize,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    pub code_predictor_config: CodePredictorConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEncoderConfig {
    pub enc_dim: usize,
    pub sample_rate: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TtsConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub assistant_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub talker_config: TalkerConfig,
    pub speaker_encoder_config: SpeakerEncoderConfig,
}

impl Qwen3TtsConfig {
    pub fn is_qwen3_tts(&self) -> bool {
        self.model_type == "qwen3_tts"
            || self
                .architectures
                .iter()
                .any(|a| a.contains("Qwen3TTS") || a.contains("Qwen3Tts"))
    }
}

// ---- Codec (speech_tokenizer) ----------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecDecoderConfig {
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub head_dim: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub semantic_codebook_size: usize,
    pub layer_scale_initial_scale: f64,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TtsCodecConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub encoder_valid_num_quantizers: usize,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub decoder_config: CodecDecoderConfig,
}
