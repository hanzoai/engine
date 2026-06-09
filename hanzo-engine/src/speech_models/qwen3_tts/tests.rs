use std::time::Instant;

use hanzo_ml::{DType, Device, IndexOp, Result, Shape, Tensor};
use hanzo_nn::var_builder::SimpleBackend;
use hanzo_nn::Init;
use hanzo_quant::ShardedSafeTensors;

use super::config::{
    CodePredictorConfig, CodecConfig, CodecDecoderConfig, Qwen3TtsConfig, SpeakerEncoderConfig,
    TalkerConfig,
};
use super::model::Qwen3TtsModel;
use super::Qwen3TtsPipeline;
use crate::speech_models::SpeechGenerationConfig;

const WEIGHT_STDEV: f64 = 0.02;

/// On-demand random-weight backend: returns a small-magnitude normal tensor for any requested
/// name/shape so a full model can be instantiated on CPU without real safetensors. Keeping the
/// stdev tiny keeps the deep codec conv stack numerically stable end-to-end.
struct RandBackend;

impl SimpleBackend for RandBackend {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        _dev: &Device,
    ) -> Result<Tensor> {
        // Weights are fetched with the default `Init::Const(0.0)` hint, so the hint can't pick a
        // sensible init. Decide by the full dotted path: norm scales -> ones (identity), biases ->
        // zeros, everything else -> small normal noise so the forward stays finite and non-zero.
        if name.contains("norm") {
            return Tensor::ones(s, dtype, &Device::Cpu);
        }
        if name.ends_with("bias") {
            return Tensor::zeros(s, dtype, &Device::Cpu);
        }
        Tensor::randn(0f32, WEIGHT_STDEV as f32, s, &Device::Cpu)?.to_dtype(dtype)
    }

    fn get_unchecked(&self, _name: &str, dtype: DType, _dev: &Device) -> Result<Tensor> {
        Tensor::zeros(1, dtype, &Device::Cpu)
    }

    fn contains_tensor(&self, _name: &str) -> bool {
        true
    }
}

fn tiny_talker_config() -> TalkerConfig {
    TalkerConfig {
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        num_code_groups: 4,
        vocab_size: 128,
        text_vocab_size: 256,
        text_hidden_size: 0,
        head_dim: 16,
        rope_theta: 10000.0,
        rms_norm_eps: 1e-6,
        max_position_embeddings: 512,
        position_id_per_seconds: 0,
        rope_scaling: None,
        code_predictor_config: CodePredictorConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_code_groups: 4,
            vocab_size: 128,
            head_dim: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            max_position_embeddings: 64,
        },
        codec_bos_id: 1,
        codec_eos_token_id: 127,
        codec_pad_id: 2,
        codec_think_id: 0,
        codec_nothink_id: 0,
        codec_think_bos_id: 0,
        codec_think_eos_id: 0,
    }
}

fn tiny_codec_config() -> CodecConfig {
    CodecConfig {
        model_type: "qwen3_tts_codec".to_string(),
        input_sample_rate: 24000,
        output_sample_rate: 24000,
        decode_upsample_rate: 8,
        encode_downsample_rate: 8,
        encoder_valid_num_quantizers: 4,
        decoder_config: CodecDecoderConfig {
            latent_dim: 32,
            codebook_dim: 32,
            codebook_size: 128,
            decoder_dim: 32,
            hidden_size: 32,
            intermediate_size: 64,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            num_quantizers: 4,
            num_semantic_quantizers: 1,
            semantic_codebook_size: 128,
            head_dim: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            sliding_window: None,
            max_position_embeddings: 512,
            upsample_rates: vec![2, 2],
            upsampling_ratios: vec![2, 2],
            vector_quantization_hidden_dimension: 32,
            layer_scale_initial_scale: 1.0,
        },
    }
}

fn tiny_config() -> Qwen3TtsConfig {
    Qwen3TtsConfig {
        model_type: "qwen3_tts".to_string(),
        talker_config: tiny_talker_config(),
        speaker_encoder_config: SpeakerEncoderConfig {
            enc_dim: 64,
            sample_rate: 24000,
        },
        tts_bos_token_id: 1,
        tts_eos_token_id: 0,
        tts_pad_token_id: 2,
        im_start_token_id: 0,
        im_end_token_id: 0,
        assistant_token_id: 0,
    }
}

fn build_pipeline() -> Qwen3TtsPipeline {
    let dtype = DType::F32;
    let vb = ShardedSafeTensors::wrap(Box::new(RandBackend), dtype, Device::Cpu);
    let codec_vb = ShardedSafeTensors::wrap(Box::new(RandBackend), dtype, Device::Cpu);
    let cfg = tiny_config();
    let codec_cfg = tiny_codec_config();
    let model = Qwen3TtsModel::new(&cfg, &codec_cfg, vb, codec_vb).unwrap();
    Qwen3TtsPipeline {
        model,
        cfg,
        device: Device::Cpu,
        dtype,
    }
}

fn rms(pcm: &[f32]) -> f32 {
    if pcm.is_empty() {
        return 0.0;
    }
    let s: f64 = pcm.iter().map(|x| (*x as f64) * (*x as f64)).sum();
    (s / pcm.len() as f64).sqrt() as f32
}

/// The KV-cache is only valid if incremental decoding reproduces the logits a full-prefix recompute
/// would give. Build a sequence, decode it token-by-token through the cache, and compare the final
/// row of codebook-0 logits against a single full forward over the whole sequence.
#[test]
fn kv_cache_matches_full_recompute() {
    let pipe = build_pipeline();
    let bos = pipe.cfg.talker_config.codec_bos_id;
    let ids: Vec<u32> = vec![bos, 3, 7, 11, 5, 9, 2, 8];
    let embed = |seq: &[u32]| -> Tensor {
        let t = Tensor::new(seq, &Device::Cpu).unwrap();
        pipe.model.talker.embed_codec(&t).unwrap().unsqueeze(0).unwrap()
    };

    let full_in = embed(&ids);
    let mut full_cache = pipe.model.talker.make_cache();
    let mask = super::model::causal_mask(ids.len(), DType::F32, &Device::Cpu).unwrap();
    let (_, full_logits) = pipe
        .model
        .talker
        .forward(&full_in, &[0], Some(&mask), &mut full_cache)
        .unwrap();
    let full_last: Vec<f32> = full_logits
        .i((0, full_logits.dim(1).unwrap() - 1, ..))
        .unwrap()
        .to_vec1()
        .unwrap();

    let mut inc_cache = pipe.model.talker.make_cache();
    let mut inc_last: Vec<f32> = Vec::new();
    for (pos, &id) in ids.iter().enumerate() {
        let step = embed(&[id]);
        let (_, logits) = pipe
            .model
            .talker
            .forward(&step, &[pos], None, &mut inc_cache)
            .unwrap();
        inc_last = logits.i((0, 0, ..)).unwrap().to_vec1().unwrap();
    }

    let max_diff = full_last
        .iter()
        .zip(inc_last.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("qwen3-tts kv-cache vs full-recompute max logit diff = {max_diff:e}");
    assert!(max_diff < 1e-3, "kv-cache logits diverge from full recompute: {max_diff}");
}

#[test]
fn coherent_audio_and_finite() {
    let pipe = build_pipeline();
    let gen = SpeechGenerationConfig::Qwen3Tts {
        max_tokens: Some(24),
        temperature: 0.9,
        top_p: 0.95,
        top_k: Some(50),
        repetition_penalty: 1.05,
    };
    let mut chunks = 0usize;
    let out = pipe
        .generate_streaming("Hello, this is a test", &gen, |c| {
            assert!(c.iter().all(|x| x.is_finite()), "streamed chunk has non-finite samples");
            chunks += 1;
        })
        .unwrap();
    assert!(!out.pcm.is_empty(), "no audio generated");
    assert!(out.pcm.iter().all(|x| x.is_finite()), "pcm has non-finite samples");
    let r = rms(&out.pcm);
    let maxabs = out.pcm.iter().map(|x| x.abs()).fold(0f32, f32::max);
    assert!(r.is_finite() && r > 0.0, "rms not sane: {r}");
    assert!(chunks > 0, "streaming callback never fired");
    eprintln!(
        "qwen3-tts: samples={} rms={:e} maxabs={:e} chunks_streamed={}",
        out.pcm.len(),
        r,
        maxabs,
        chunks
    );
}

/// Compares per-token decode latency of the KV-cached path against a forced no-cache path that
/// rebuilds a fresh cache and re-runs the full prefix each step (the milestone-1 O(n^2) behavior).
#[test]
fn latency_kv_cache_vs_recompute() {
    let pipe = build_pipeline();
    let n_tokens = 64usize;

    let warm = pipe.model.talker.make_cache();
    drop(warm);

    // Cached: prefill once, then single-token decode steps reusing the cache.
    let bytes: Vec<u32> = "Hello, this is a test".bytes().map(|b| b as u32).collect();
    let text_ids = Tensor::new(bytes.as_slice(), &Device::Cpu)
        .unwrap()
        .unsqueeze(0)
        .unwrap();
    let prefill = pipe.prefill_embeds(&text_ids).unwrap();
    let prefill_len = prefill.dim(1).unwrap();
    let mut cache = pipe.model.talker.make_cache();
    let mask = super::model::causal_mask(prefill_len, DType::F32, &Device::Cpu).unwrap();
    let _ = pipe
        .model
        .talker
        .forward(&prefill, &[0], Some(&mask), &mut cache)
        .unwrap();
    let bos = Tensor::new(&[pipe.cfg.talker_config.codec_bos_id], &Device::Cpu).unwrap();
    let step_embed = pipe.model.talker.embed_codec(&bos).unwrap().unsqueeze(0).unwrap();
    let t0 = Instant::now();
    for _ in 0..n_tokens {
        let offset = cache[0].current_seq_len();
        let _ = pipe
            .model
            .talker
            .forward(&step_embed, &[offset], None, &mut cache)
            .unwrap();
    }
    let cached_us = t0.elapsed().as_micros() as f64 / n_tokens as f64;

    // No-cache: every step rebuilds a fresh cache and re-runs the whole growing prefix.
    let mut seq = prefill.clone();
    let t1 = Instant::now();
    for _ in 0..n_tokens {
        seq = Tensor::cat(&[&seq, &step_embed], 1).unwrap();
        let mut fresh = pipe.model.talker.make_cache();
        let m = super::model::causal_mask(seq.dim(1).unwrap(), DType::F32, &Device::Cpu).unwrap();
        let _ = pipe
            .model
            .talker
            .forward(&seq, &[0], Some(&m), &mut fresh)
            .unwrap();
    }
    let recompute_us = t1.elapsed().as_micros() as f64 / n_tokens as f64;

    eprintln!(
        "qwen3-tts latency/token over {n_tokens} steps (prefix_len0={prefill_len}): kv-cache={cached_us:.1}us  recompute={recompute_us:.1}us  speedup={:.2}x",
        recompute_us / cached_us
    );
    assert!(cached_us < recompute_us, "kv-cache not faster than full recompute");
}
