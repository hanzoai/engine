//! Qwen3-Omni-MoE: the fused understand → think → speak multimodal model.
//!
//! `model_type = "qwen3_omni_moe"`, HF class `Qwen3OmniMoeForConditionalGeneration`.

pub mod audio_tower;
pub mod code2wav;
pub mod config;
pub mod talker;
pub mod thinker;

pub use config::Qwen3OmniConfig;

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
