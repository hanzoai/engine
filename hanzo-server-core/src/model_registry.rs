//! Zen SKU -> HF repo -> arch registry.
//!
//! When a client sends `model: "zen5-pro"` over `/v1/messages` or
//! `/v1/chat/completions`, the engine looks up that SKU here to decide which
//! Hugging Face repo to fetch and which backend arch (deepseek-v4-flash,
//! qwen3-moe, qwen3-vl, ...) to load.
//!
//! Three things live here:
//!   1. The canonical Zen SKU set served by zen-gateway and listed in
//!      `pricing/src/models.mjs::zenCatalog`.
//!   2. The HF source repo (under `zenlm/`) for each SKU. Text/embedding/vision
//!      repos are research's verified-to-resolve artifacts (GGUF mirrors carry a
//!      `-gguf` suffix); ASR/TTS repos follow canon's published-artifact names.
//!   3. The arch kind:
//!        - `Supported(NormalLoaderType)` — text models loadable by the normal
//!          (text) loader.
//!        - `SupportedVision(MultimodalLoaderType)` — vision models loadable by
//!          the multimodal pipeline (qwen3-vl is ported: `Qwen3VLLoader` /
//!          `Qwen3VLMoELoader`; qwen3-omni is ported: `Qwen3OmniLoader`).
//!        - `SupportedAudio(AsrLoaderType)` — speech models loadable by the ASR
//!          pipeline (qwen3-asr is ported: `speech_models/qwen3_asr`).
//!        - `Unsupported(name)` — arch not yet ported (qwen3-tts,
//!          deepseek-v4-flash/pro); handlers return a clean 501.
//!
//! Adding a new SKU is a one-line change in `zen_sku_table()`. When a new arch
//! lands in `hanzo-engine`, flip its `Unsupported` to the matching `Supported*`.

use std::collections::HashMap;
use std::sync::OnceLock;

/// Backend modality. Matches the gateway/pricing modality taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Modality {
    Text,
    Vision,
    Audio,
    Embedding,
}

/// Which arch loads this SKU, and via which loader family.
///
/// Each `Supported*` carries the loader name string the engine accepts:
/// `Supported` = `NormalLoaderType` (text), `SupportedVision` =
/// `MultimodalLoaderType`, `SupportedAudio` = the ASR loader. `Unsupported`
/// carries a free-form placeholder so handlers can produce a 501 with a useful
/// message and a pointer to the planned arch name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchKind {
    Supported(&'static str),
    SupportedVision(&'static str),
    SupportedAudio(&'static str),
    Unsupported(&'static str),
}

impl ArchKind {
    pub fn name(&self) -> &'static str {
        match self {
            ArchKind::Supported(n)
            | ArchKind::SupportedVision(n)
            | ArchKind::SupportedAudio(n)
            | ArchKind::Unsupported(n) => n,
        }
    }

    /// True if the engine can load this arch (via any loader family).
    pub fn is_supported(&self) -> bool {
        !matches!(self, ArchKind::Unsupported(_))
    }
}

/// A single Zen SKU entry.
#[derive(Debug, Clone)]
pub struct ZenSku {
    /// User-facing SKU name (e.g. `zen5-pro`). What clients put in `model`.
    pub sku: &'static str,
    /// Hugging Face repo id (e.g. `zenlm/zen-5-pro-gguf`).
    pub hf_repo: &'static str,
    /// Backend arch.
    pub arch: ArchKind,
    /// Quant variant when applicable (`IQ2_XXS`, `Q4_K_M`, ...), else `None`.
    pub quant: Option<&'static str>,
    pub modality: Modality,
}

/// Canonical Zen SKU table.
///
/// Source of truth is the intersection of zen-gateway `config.yaml::model_list`
/// and pricing `models.mjs::zenCatalog`. Add/edit rows here, then run
/// `cargo check -p hanzo-server-core`. HF repo ids are the actual published
/// artifacts (verified 2026-06-06 against huggingface.co/zenlm).
fn zen_sku_table() -> &'static [ZenSku] {
    &[
        // ---- Zen5 ladder (canonical) ---------------------------------------
        // zen5-nano-*: Zen VL dense edge tier (Qwen3-VL arch). The multimodal
        // pipeline (Qwen3VLLoader) is ported, so these are SupportedVision.
        ZenSku {
            sku: "zen5-nano-0.8B",
            hf_repo: "zenlm/zen-5-nano-0.8B-gguf",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        ZenSku {
            sku: "zen5-nano-2B",
            hf_repo: "zenlm/zen-5-nano-2B-gguf",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        ZenSku {
            sku: "zen5-nano-4B",
            hf_repo: "zenlm/zen-5-nano-4B-gguf",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        ZenSku {
            sku: "zen5-nano-9B",
            hf_repo: "zenlm/zen-5-nano-9B-gguf",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        // Text-only Zen5 ladder. qwen3 / qwen3moe are in tree. GGUF mirrors
        // carry the `-gguf` suffix.
        ZenSku {
            sku: "zen5-flash",
            hf_repo: "zenlm/zen-5-flash-gguf",
            arch: ArchKind::Supported("qwen3"),
            quant: None,
            modality: Modality::Text,
        },
        ZenSku {
            sku: "zen5-mini",
            hf_repo: "zenlm/zen-5-mini-gguf",
            arch: ArchKind::Supported("qwen3moe"),
            quant: None,
            modality: Modality::Text,
        },
        ZenSku {
            sku: "zen5",
            hf_repo: "zenlm/zen-5-gguf",
            arch: ArchKind::Supported("qwen3moe"),
            quant: None,
            modality: Modality::Text,
        },
        ZenSku {
            sku: "zen5-coder",
            hf_repo: "zenlm/zen-5-coder-gguf",
            arch: ArchKind::Supported("qwen3moe"),
            quant: None,
            modality: Modality::Text,
        },
        // Zen5 Pro/Max ride on DeepSeek V4 (Flash / Pro). DS4 is not on main;
        // flip to Supported once the deepseek-v4 arch (hanzo-engine zen5.rs)
        // lands in hanzo-engine.
        ZenSku {
            sku: "zen5-pro",
            hf_repo: "zenlm/zen-5-pro-gguf",
            arch: ArchKind::Unsupported("deepseek-v4-flash"),
            quant: Some("IQ2_XXS"),
            modality: Modality::Text,
        },
        ZenSku {
            sku: "zen5-max",
            hf_repo: "zenlm/zen-5-max-gguf",
            arch: ArchKind::Unsupported("deepseek-v4-pro"),
            quant: Some("Q4_K_M"),
            modality: Modality::Text,
        },
        // Zen5 embeddings. qwen3 backbone is in tree; the embedding head is
        // served by `hanzo-engine::pipeline::embedding`. Weights live under the
        // `zen-embedding-*` repos (GGUF mirrors are `-GGUF`).
        ZenSku {
            sku: "zen5-embedding-0.6B",
            hf_repo: "zenlm/zen-embedding-0.6B-GGUF",
            arch: ArchKind::Supported("qwen3"),
            quant: None,
            modality: Modality::Embedding,
        },
        ZenSku {
            sku: "zen5-embedding-4B",
            hf_repo: "zenlm/zen-embedding-4B",
            arch: ArchKind::Supported("qwen3"),
            quant: None,
            modality: Modality::Embedding,
        },
        ZenSku {
            sku: "zen5-embedding-8B",
            hf_repo: "zenlm/zen-embedding-8B-GGUF",
            arch: ArchKind::Supported("qwen3"),
            quant: None,
            modality: Modality::Embedding,
        },
        // Zen4 generation (zen4, zen4-pro, zen4-max, zen4.1, zen4-mini,
        // zen4-ultra, zen4-thinking, zen4-coder, zen4-coder-flash,
        // zen4-coder-pro) sunset 2026-05-30 with the zenlm/zen4* HF mirrors.
        // Routing still works via the gateway's Fireworks aliases for any
        // legacy callers, but the engine no longer claims to be able to load
        // them directly. Use the zen5 ladder instead.

        // ---- Zen3 family (multimodal + specialty) --------------------------
        // qwen3-omni IS ported (Qwen3OmniLoader / vision_models::qwen3_omni): the multimodal
        // pipeline serves its Thinker text path; SupportedVision.
        ZenSku {
            sku: "zen3-omni",
            hf_repo: "zenlm/zen-omni",
            arch: ArchKind::SupportedVision("qwen3-omni"),
            quant: None,
            modality: Modality::Vision,
        },
        // qwen3-vl IS ported (Qwen3VLLoader / Qwen3VLMoELoader). Real weights
        // live under `zen3-vl` and the `zen-vl-*-instruct` repos.
        ZenSku {
            sku: "zen3-vl",
            hf_repo: "zenlm/zen3-vl",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        // zen3-vl-2B: no public 2B weights yet (registry repo 401); arch is
        // supported, so this will load once a 2B repo is published.
        ZenSku {
            sku: "zen3-vl-2B",
            hf_repo: "zenlm/zen-vl-2b",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        ZenSku {
            sku: "zen3-vl-8B",
            hf_repo: "zenlm/zen-vl-8b-instruct",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        // zen3-vl-32B: closest published artifact is the 30B instruct repo.
        ZenSku {
            sku: "zen3-vl-32B",
            hf_repo: "zenlm/zen-vl-30b-instruct",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        // zen3-vl-235B-A22B: no public weights yet (registry repo 401).
        ZenSku {
            sku: "zen3-vl-235B-A22B",
            hf_repo: "zenlm/zen-vl-235b-a22b",
            arch: ArchKind::SupportedVision("qwen3-vl"),
            quant: None,
            modality: Modality::Vision,
        },
        // zen3-vl-reranker-{2B,8B}, zen3-vl-embedding-{2B,8B}, and
        // zen3-web-{8B,14B,32B} were virtual SKUs with no HF weights — removed
        // 2026-05-30. The canonical zen3-vl-* size variants above remain.
        ZenSku {
            sku: "zen3-nano",
            hf_repo: "zenlm/zen-nano-0.6b",
            arch: ArchKind::Supported("llama"),
            quant: None,
            modality: Modality::Text,
        },
        ZenSku {
            sku: "zen3-guard",
            hf_repo: "zenlm/zen-guard",
            arch: ArchKind::Supported("mixtral"),
            quant: None,
            modality: Modality::Text,
        },
        // Zen3 ASR: qwen3-asr IS ported (speech_models/qwen3_asr,
        // AsrLoaderType::Qwen3Asr). Repos follow canon's published `zen-asr-*`
        // artifact names.
        ZenSku {
            sku: "zen3-asr",
            hf_repo: "zenlm/zen-asr-1.7b",
            arch: ArchKind::SupportedAudio("qwen3-asr"),
            quant: None,
            modality: Modality::Audio,
        },
        ZenSku {
            sku: "zen3-asr-0.6B",
            hf_repo: "zenlm/zen-asr-0.6b",
            arch: ArchKind::SupportedAudio("qwen3-asr"),
            quant: None,
            modality: Modality::Audio,
        },
        ZenSku {
            sku: "zen3-asr-aligner",
            hf_repo: "zenlm/zen-asr-aligner-0.6b",
            arch: ArchKind::SupportedAudio("qwen3-asr"),
            quant: None,
            modality: Modality::Audio,
        },
        // Zen3 TTS: qwen3-tts not yet ported; Unsupported. Repos follow canon's
        // published `zen-tts-*` artifact names.
        ZenSku {
            sku: "zen3-tts",
            hf_repo: "zenlm/zen-tts-1.7b",
            arch: ArchKind::Unsupported("qwen3-tts"),
            quant: None,
            modality: Modality::Audio,
        },
        ZenSku {
            sku: "zen3-tts-0.6B",
            hf_repo: "zenlm/zen-tts-0.6b",
            arch: ArchKind::Unsupported("qwen3-tts"),
            quant: None,
            modality: Modality::Audio,
        },
        ZenSku {
            sku: "zen3-tts-voice-design",
            hf_repo: "zenlm/zen-tts-voicedesign-1.7b",
            arch: ArchKind::Unsupported("qwen3-tts"),
            quant: None,
            modality: Modality::Audio,
        },
        ZenSku {
            sku: "zen3-tts-custom-voice",
            hf_repo: "zenlm/zen-tts-customvoice-1.7b",
            arch: ArchKind::Unsupported("qwen3-tts"),
            quant: None,
            modality: Modality::Audio,
        },
        // Zen3 embeddings (text-only, in-tree). zen3-embedding-medium /
        // zen3-embedding-small ghost aliases were sunset 2026-05-30; use the
        // zen5-embedding-{0.6B,4B,8B} ladder for new integrations.
        ZenSku {
            sku: "zen3-embedding",
            hf_repo: "zenlm/zen-embedding",
            arch: ArchKind::Supported("qwen3"),
            quant: None,
            modality: Modality::Embedding,
        },
    ]
}

/// SKU -> entry index. Built on first lookup.
fn index() -> &'static HashMap<&'static str, usize> {
    static IDX: OnceLock<HashMap<&'static str, usize>> = OnceLock::new();
    IDX.get_or_init(|| {
        zen_sku_table()
            .iter()
            .enumerate()
            .map(|(i, e)| (e.sku, i))
            .collect()
    })
}

/// Look up an SKU. Case-sensitive — the canonical case matches `zenCatalog`.
pub fn lookup(sku: &str) -> Option<&'static ZenSku> {
    index().get(sku).map(|&i| &zen_sku_table()[i])
}

/// Iterate every registered SKU.
pub fn all() -> &'static [ZenSku] {
    zen_sku_table()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skus_are_unique() {
        let table = zen_sku_table();
        let mut seen = std::collections::HashSet::new();
        for e in table {
            assert!(seen.insert(e.sku), "duplicate SKU `{}`", e.sku);
        }
    }

    #[test]
    fn known_skus_resolve() {
        for sku in [
            "zen5",
            "zen5-pro",
            "zen5-max",
            "zen5-flash",
            "zen5-coder",
            "zen5-embedding-8B",
            "zen3-vl",
            "zen3-asr",
            "zen3-nano",
        ] {
            assert!(lookup(sku).is_some(), "missing SKU `{sku}`");
        }
    }

    #[test]
    fn sunset_skus_are_absent() {
        // Zen4 generation + virtual zen3 aliases were removed 2026-05-30.
        for sku in [
            "zen4",
            "zen4-pro",
            "zen4-max",
            "zen4.1",
            "zen4-mini",
            "zen4-ultra",
            "zen4-thinking",
            "zen4-coder",
            "zen4-coder-flash",
            "zen4-coder-pro",
            "zen5-ultra",
            "zen3-vl-reranker-2B",
            "zen3-vl-reranker-8B",
            "zen3-vl-embedding-2B",
            "zen3-vl-embedding-8B",
            "zen3-web-8B",
            "zen3-web-14B",
            "zen3-web-32B",
            "zen3-embedding-small",
            "zen3-embedding-medium",
        ] {
            assert!(
                lookup(sku).is_none(),
                "sunset SKU `{sku}` still in registry"
            );
        }
    }

    #[test]
    fn unsupported_arches_are_explicit() {
        // DeepSeek-V4 (zen5-pro/max) and qwen3-tts are not yet ported.
        let pro = lookup("zen5-pro").unwrap();
        assert!(!pro.arch.is_supported());
        assert_eq!(pro.arch.name(), "deepseek-v4-flash");
        // qwen3-omni IS ported now (Qwen3OmniLoader) — supported via the multimodal pipeline.
        let omni = lookup("zen3-omni").unwrap();
        assert!(omni.arch.is_supported());
        assert_eq!(omni.arch.name(), "qwen3-omni");
        let tts = lookup("zen3-tts").unwrap();
        assert!(!tts.arch.is_supported());
        assert_eq!(tts.arch.name(), "qwen3-tts");
    }

    #[test]
    fn ported_multimodal_and_asr_are_supported() {
        // qwen3-vl (Qwen3VLLoader) and qwen3-asr (speech_models/qwen3_asr) ARE
        // ported — they load via the multimodal / ASR pipelines, not the text
        // loader.
        let vl = lookup("zen3-vl").unwrap();
        assert!(vl.arch.is_supported());
        assert_eq!(vl.arch, ArchKind::SupportedVision("qwen3-vl"));
        let asr = lookup("zen3-asr").unwrap();
        assert!(asr.arch.is_supported());
        assert_eq!(asr.arch, ArchKind::SupportedAudio("qwen3-asr"));
    }

    #[test]
    fn supported_arch_names_match_loaders() {
        for e in all() {
            match &e.arch {
                // Text loaders: NormalLoaderType names.
                ArchKind::Supported(name) => match *name {
                    "qwen3" | "qwen3moe" | "deepseekv3" | "glm4moe" | "llama" | "mixtral"
                    | "gpt_oss" => {}
                    other => panic!(
                        "SKU `{}` uses text arch `{}` not in NormalLoaderType",
                        e.sku, other
                    ),
                },
                // Multimodal loaders: MultimodalLoaderType arch names.
                ArchKind::SupportedVision(name) => assert!(
                    matches!(*name, "qwen3-vl" | "qwen3-omni"),
                    "SKU `{}` uses unexpected vision arch `{}`",
                    e.sku,
                    name
                ),
                // ASR loaders.
                ArchKind::SupportedAudio(name) => assert_eq!(
                    *name, "qwen3-asr",
                    "SKU `{}` uses unexpected audio arch `{}`",
                    e.sku, name
                ),
                ArchKind::Unsupported(_) => {}
            }
        }
    }
}
