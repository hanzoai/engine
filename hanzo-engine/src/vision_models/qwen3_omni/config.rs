#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Configuration for `Qwen3OmniMoeForConditionalGeneration` (`model_type = "qwen3_omni_moe"`).
//!
//! Omni is *understand → think → speak*: a shared **Thinker** (Qwen3-MoE text decoder + a
//! Qwen3-VL-style vision tower + a Whisper-style audio tower) consumes interleaved
//! text/image/video/audio and emits text logits plus a hidden-state stream; the **Talker**
//! (a smaller Qwen3-MoE decoder + an MTP code-predictor) renders that stream to neural-codec
//! codes, which **Code2Wav** (a RVQ + ConvNeXt/SnakeBeta vocoder) turns into a 24 kHz waveform.
//!
//! The three top-level weight namespaces — `thinker.*`, `talker.*`, `code2wav.*` — map one-to-one
//! onto the three sub-configs here. Field names mirror the published `config.json` so a checkpoint
//! deserializes directly.

use serde::Deserialize;

use hanzo_quant::QuantizedConfig;

use crate::layers::Activation;
use crate::serde_default_fn;

// Re-use the Qwen3-VL vision tower config — the omni `qwen3_omni_moe_vision_encoder` is the same
// ViT (depth/heads/patch/deepstack indices) plus absolute position embeddings.
pub use crate::vision_models::qwen3_vl::config::VisionConfig;

serde_default_fn!(bool, default_true, true);
serde_default_fn!(bool, default_false, false);
serde_default_fn!(Vec<usize>, default_empty_usize, Vec::new());
serde_default_fn!(f64, default_rope_theta_1e6, 1_000_000.0);
serde_default_fn!(f64, default_rms_eps_1e6, 1e-6);
serde_default_fn!(usize, default_head_dim_128, 128);

/// mRoPE / rope scaling block. `mrope_section` partitions the rotary dims across the
/// (temporal, height, width) position axes for multimodal rotary embeddings.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_false")]
    pub interleaved: bool,
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
}

/// `qwen3_omni_moe_text` — the Thinker decoder: a full-attention Qwen3-MoE with QK-norm.
/// (No shared expert: `shared_expert_intermediate_size == 0`.)
#[derive(Debug, Clone, Deserialize)]
pub struct OmniTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim_128")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps_1e6")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta_1e6")]
    pub rope_theta: f64,
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_empty_usize")]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    #[serde(default = "default_false")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
    /// In-checkpoint quantization (FP8 / GPTQ / AWQ / …) for the Thinker's quantizable linears.
    /// `None` for a full-precision checkpoint; runtime ISQ is orthogonal and applies on top of an
    /// unquantized checkpoint. HF places this at the config top level, so [`Qwen3OmniConfig::new`]'s
    /// caller propagates the top-level field here before the Thinker is built.
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

serde_default_fn!(usize, default_one, 1);

fn default_hidden_act() -> Activation {
    Activation::Silu
}

impl OmniTextConfig {
    /// True if layer `idx` uses MoE (vs a dense MLP).
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        !self.mlp_only_layers.contains(&idx)
            && self.num_experts > 0
            && (idx + 1) % self.decoder_sparse_step == 0
    }
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .map(|r| r.mrope_section.clone())
            .unwrap_or_default()
    }
}

/// `qwen3_omni_moe_audio_encoder` — Whisper-style mel→hidden encoder feeding the Thinker.
#[derive(Debug, Clone, Deserialize)]
pub struct OmniAudioConfig {
    pub d_model: usize,
    pub encoder_layers: usize,
    pub encoder_attention_heads: usize,
    pub encoder_ffn_dim: usize,
    pub num_mel_bins: usize,
    pub output_dim: usize,
    #[serde(default = "default_max_source_positions")]
    pub max_source_positions: usize,
    #[serde(default = "default_n_window")]
    pub n_window: usize,
    #[serde(default = "default_n_window_infer")]
    pub n_window_infer: usize,
    #[serde(default = "default_conv_chunksize")]
    pub conv_chunksize: usize,
    #[serde(default = "default_downsample_hidden")]
    pub downsample_hidden_size: usize,
    #[serde(default = "default_audio_act")]
    pub activation_function: Activation,
    #[serde(default = "default_false")]
    pub scale_embedding: bool,
}

serde_default_fn!(usize, default_max_source_positions, 1500);
serde_default_fn!(usize, default_n_window, 50);
serde_default_fn!(usize, default_n_window_infer, 800);
serde_default_fn!(usize, default_conv_chunksize, 500);
serde_default_fn!(usize, default_downsample_hidden, 480);
fn default_audio_act() -> Activation {
    Activation::Gelu
}

impl OmniAudioConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }
}

/// Aggregate Thinker config: `thinker.*` weight namespace.
#[derive(Debug, Clone, Deserialize)]
pub struct ThinkerConfig {
    pub text_config: OmniTextConfig,
    pub vision_config: VisionConfig,
    pub audio_config: OmniAudioConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub audio_token_id: u32,
    pub audio_start_token_id: u32,
    pub audio_end_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub user_token_id: u32,
    #[serde(default = "default_position_id_per_seconds")]
    pub position_id_per_seconds: usize,
    #[serde(default = "default_seconds_per_chunk")]
    pub seconds_per_chunk: usize,
}

serde_default_fn!(usize, default_position_id_per_seconds, 13);
serde_default_fn!(usize, default_seconds_per_chunk, 2);

/// `qwen3_omni_moe_text` flavour used by the Talker — a smaller Qwen3-MoE *with* a shared expert.
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim_128")]
    pub head_dim: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_eps_1e6")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta_1e6")]
    pub rope_theta: f64,
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_true")]
    pub norm_topk_prob: bool,
    #[serde(default = "default_empty_usize")]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,
    #[serde(default)]
    pub rope_scaling: Option<RopeScaling>,
}

impl TalkerTextConfig {
    pub fn is_moe_layer(&self, idx: usize) -> bool {
        !self.mlp_only_layers.contains(&idx)
            && self.num_experts > 0
            && (idx + 1) % self.decoder_sparse_step == 0
    }
    pub fn has_shared_expert(&self) -> bool {
        self.shared_expert_intermediate_size > 0
    }
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .map(|r| r.mrope_section.clone())
            .unwrap_or_default()
    }
}

/// `qwen3_omni_moe_talker_code_predictor` — the dense MTP head that predicts codebooks 1..N
/// from the Talker hidden state.
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim_128")]
    pub head_dim: usize,
    pub num_code_groups: usize,
    pub vocab_size: usize,
    #[serde(default = "default_rms_eps_1e6")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta_1e6")]
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

/// Aggregate Talker config: `talker.*` weight namespace.
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub text_config: TalkerTextConfig,
    pub code_predictor_config: CodePredictorConfig,
    pub num_code_groups: usize,
    /// Width of the Thinker hidden state fed through `hidden_projection`.
    pub thinker_hidden_size: usize,
    /// Which Thinker decoder layer's hidden state the Talker consumes.
    pub accept_hidden_layer: usize,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    #[serde(default)]
    pub codec_nothink_id: u32,
    #[serde(default)]
    pub codec_think_bos_id: u32,
    #[serde(default)]
    pub codec_think_eos_id: u32,
    #[serde(default = "default_position_id_per_seconds")]
    pub position_id_per_seconds: usize,
}

/// `code2wav_config` — the neural-codec waveform decoder (`code2wav.*`).
#[derive(Debug, Clone, Deserialize)]
pub struct Code2WavConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub decoder_dim: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub semantic_codebook_size: usize,
    pub vector_quantization_hidden_dimension: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_initial_scale: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default = "default_rms_eps_1e5")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta_1e4")]
    pub rope_theta: f64,
    #[serde(default = "default_max_pos_8000")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_audio_act")]
    pub hidden_act: Activation,
}

serde_default_fn!(f64, default_layer_scale, 0.01);
serde_default_fn!(f64, default_rms_eps_1e5, 1e-5);
serde_default_fn!(f64, default_rope_theta_1e4, 10_000.0);
serde_default_fn!(usize, default_max_pos_8000, 8000);

/// Top-level `qwen3_omni_moe` config.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3OmniConfig {
    pub thinker_config: ThinkerConfig,
    pub talker_config: TalkerConfig,
    pub code2wav_config: Code2WavConfig,
    #[serde(default = "default_true")]
    pub enable_audio_output: bool,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    #[serde(default)]
    pub system_token_id: u32,
    #[serde(default)]
    pub user_token_id: u32,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    /// Top-level in-checkpoint quantization config (FP8 / GPTQ / …). Takes precedence over and is
    /// propagated into [`OmniTextConfig::quantization_config`] when the Thinker is constructed, so a
    /// pre-quantized Omni checkpoint loads the Thinker linears through the quantized path.
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}
