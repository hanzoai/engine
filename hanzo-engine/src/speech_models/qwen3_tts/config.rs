use serde::{Deserialize, Serialize};

fn default_rope_theta() -> f64 {
    1_000_000.0
}

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_head_dim() -> usize {
    128
}

/// The sub-talker (MTP) that predicts codebooks 1..num_code_groups from the talker hidden state.
/// In the real checkpoint this is `talker.code_predictor.{model,lm_head}` with per-group embeddings
/// and per-group linear heads. `small_to_mtp_projection` is absent because hidden_size == talker hidden_size.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_code_groups: usize,
    pub vocab_size: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub max_position_embeddings: usize,
}

impl CodePredictorConfig {
    pub fn head_dim(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(rename = "type", default)]
    pub rope_type: String,
}

/// The autoregressive talker backbone. Note `text_hidden_size` (2048) differs from `hidden_size`
/// (1024): text token embeddings live in 2048-d and are projected down to 1024-d by `text_projection`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_code_groups: usize,
    pub vocab_size: usize,
    pub text_vocab_size: usize,
    pub text_hidden_size: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub position_id_per_seconds: usize,
    pub rope_scaling: Option<RopeScaling>,
    pub code_predictor_config: CodePredictorConfig,

    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    #[serde(default)]
    pub codec_think_id: u32,
    #[serde(default)]
    pub codec_nothink_id: u32,
    #[serde(default)]
    pub codec_think_bos_id: u32,
    #[serde(default)]
    pub codec_think_eos_id: u32,
}

impl TalkerConfig {
    pub fn head_dim(&self) -> usize {
        if self.head_dim != 0 {
            self.head_dim
        } else {
            self.hidden_size / self.num_attention_heads
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerEncoderConfig {
    pub enc_dim: usize,
    pub sample_rate: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Qwen3TtsConfig {
    pub model_type: String,
    pub talker_config: TalkerConfig,
    pub speaker_encoder_config: SpeakerEncoderConfig,

    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    #[serde(default)]
    pub im_start_token_id: u32,
    #[serde(default)]
    pub im_end_token_id: u32,
    #[serde(default)]
    pub assistant_token_id: u32,
}

fn default_head_dim_64() -> usize {
    64
}

fn default_codec_rms_eps() -> f64 {
    1e-5
}

fn default_codec_rope_theta() -> f64 {
    10000.0
}

/// The speech-tokenizer (codec) decoder config (`speech_tokenizer/config.json` -> `decoder_config`).
/// This drives the SplitRVQ + pre_transformer + upsample + conv decoder that turns codec indices
/// into a 24 kHz waveform.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecDecoderConfig {
    pub latent_dim: usize,
    pub codebook_dim: usize,
    pub codebook_size: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub num_hidden_layers: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub semantic_codebook_size: usize,
    #[serde(default = "default_head_dim_64")]
    pub head_dim: usize,
    #[serde(default = "default_codec_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_codec_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_position_embeddings: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub vector_quantization_hidden_dimension: usize,
    #[serde(default)]
    pub layer_scale_initial_scale: f64,
    #[serde(default = "default_codec_act")]
    pub hidden_act: String,
}

fn default_codec_act() -> String {
    "silu".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    pub model_type: String,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub encoder_valid_num_quantizers: usize,
    pub decoder_config: CodecDecoderConfig,
}
