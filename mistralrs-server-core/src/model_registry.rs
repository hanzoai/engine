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
//!   2. The HF source repo (under `zenlm/`) for each SKU.
//!   3. The arch kind, which is either `Supported(NormalLoaderType)` or
//!      `Unsupported(name)` for archs not yet ported. The server uses the
//!      latter to return a clean 501 instead of a 500 deep in the loader.
//!
//! Adding a new SKU is a one-line change in `zen_sku_table()`. Porting a new
//! arch (qwen3-vl, qwen3-asr, ...) is out of scope for this registry; once the
//! arch lands in `mistralrs-core`, flip the `Unsupported` to `Supported`.

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

/// Which arch loads this SKU.
///
/// `Supported` carries the engine's NormalLoaderType name (the same string
/// `NormalLoaderType::from_str` accepts). `Unsupported` carries a free-form
/// placeholder so handlers can produce a 501 with a useful message and a
/// pointer to the planned arch name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchKind {
    Supported(&'static str),
    Unsupported(&'static str),
}

impl ArchKind {
    pub fn name(&self) -> &'static str {
        match self {
            ArchKind::Supported(n) | ArchKind::Unsupported(n) => n,
        }
    }

    pub fn is_supported(&self) -> bool {
        matches!(self, ArchKind::Supported(_))
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
/// `cargo check -p mistralrs-server-core`.
fn zen_sku_table() -> &'static [ZenSku] {
    &[
        // ---- Zen5 ladder (canonical) ---------------------------------------
        // zen5-nano-*: Zen VL dense edge tier. Multimodal but qwen3-vl arch is
        // not yet ported; mark Unsupported until the encoder lands.
        ZenSku { sku: "zen5-nano-0.8B", hf_repo: "zenlm/zen-5-nano-0.8b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen5-nano-2B",   hf_repo: "zenlm/zen-5-nano-2b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen5-nano-4B",   hf_repo: "zenlm/zen-5-nano-4b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen5-nano-9B",   hf_repo: "zenlm/zen-5-nano-9b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },

        // Text-only Zen5 ladder. qwen3 / qwen3moe are in tree.
        ZenSku { sku: "zen5-flash", hf_repo: "zenlm/zen-5-flash",
            arch: ArchKind::Supported("qwen3"), quant: None,
            modality: Modality::Text },
        ZenSku { sku: "zen5-mini",  hf_repo: "zenlm/zen-5-mini",
            arch: ArchKind::Supported("qwen3moe"), quant: None,
            modality: Modality::Text },
        ZenSku { sku: "zen5",       hf_repo: "zenlm/zen-5",
            arch: ArchKind::Supported("qwen3moe"), quant: None,
            modality: Modality::Text },
        ZenSku { sku: "zen5-coder", hf_repo: "zenlm/zen-5-coder",
            arch: ArchKind::Supported("qwen3moe"), quant: None,
            modality: Modality::Text },

        // Zen5 Pro/Max/Ultra ride on DeepSeek V4 (zen5 branch arch, "Zen5"
        // internally). DS4 is not on main; flip to Supported once
        // mistralrs-core/src/models/zen5.rs lands here.
        ZenSku { sku: "zen5-pro",   hf_repo: "zenlm/zen-5-pro-gguf",
            arch: ArchKind::Unsupported("deepseek-v4-flash"),
            quant: Some("IQ2_XXS"), modality: Modality::Text },
        ZenSku { sku: "zen5-max",   hf_repo: "zenlm/zen-5-max-gguf",
            arch: ArchKind::Unsupported("deepseek-v4-pro"),
            quant: Some("Q4_K_M"), modality: Modality::Text },

        // Zen5 embeddings. qwen3 backbone is in tree but the embedding head
        // is served by `mistralrs-core::pipeline::embedding`, which already
        // wraps qwen3-style backbones.
        ZenSku { sku: "zen5-embedding-0.6B", hf_repo: "zenlm/zen-5-embedding-0.6b",
            arch: ArchKind::Supported("qwen3"), quant: None,
            modality: Modality::Embedding },
        ZenSku { sku: "zen5-embedding-4B",   hf_repo: "zenlm/zen-5-embedding-4b",
            arch: ArchKind::Supported("qwen3"), quant: None,
            modality: Modality::Embedding },
        ZenSku { sku: "zen5-embedding-8B",   hf_repo: "zenlm/zen-5-embedding-8b",
            arch: ArchKind::Supported("qwen3"), quant: None,
            modality: Modality::Embedding },

        // Zen4 generation (zen4, zen4-pro, zen4-max, zen4.1, zen4-mini,
        // zen4-ultra, zen4-thinking, zen4-coder, zen4-coder-flash,
        // zen4-coder-pro) sunset 2026-05-30 with the zenlm/zen4* HF mirrors.
        // Routing still works via the gateway's Fireworks aliases for any
        // legacy callers, but the engine no longer claims to be able to load
        // them directly. Use the zen5 ladder instead.

        // ---- Zen3 family (multimodal + specialty) --------------------------
        ZenSku { sku: "zen3-omni", hf_repo: "zenlm/zen-omni",
            arch: ArchKind::Unsupported("qwen3-omni"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen3-vl", hf_repo: "zenlm/zen-vl",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen3-vl-2B",        hf_repo: "zenlm/zen-vl-2b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen3-vl-8B",        hf_repo: "zenlm/zen-vl-8b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen3-vl-32B",       hf_repo: "zenlm/zen-vl-32b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        ZenSku { sku: "zen3-vl-235B-A22B", hf_repo: "zenlm/zen-vl-235b-a22b",
            arch: ArchKind::Unsupported("qwen3-vl"), quant: None,
            modality: Modality::Vision },
        // zen3-vl-reranker-{2B,8B}, zen3-vl-embedding-{2B,8B}, and
        // zen3-web-{8B,14B,32B} were virtual SKUs with no HF weights — removed
        // 2026-05-30. The canonical zen3-vl-* size variants above remain.

        ZenSku { sku: "zen3-nano",  hf_repo: "zenlm/zen-nano-0.6b",
            arch: ArchKind::Supported("llama"), quant: None,
            modality: Modality::Text },
        ZenSku { sku: "zen3-guard", hf_repo: "zenlm/zen-guard",
            arch: ArchKind::Supported("mixtral"), quant: None,
            modality: Modality::Text },

        // Zen3 ASR / TTS — encoders not ported.
        ZenSku { sku: "zen3-asr",         hf_repo: "zenlm/zen-asr-1.7b",
            arch: ArchKind::Unsupported("qwen3-asr"), quant: None,
            modality: Modality::Audio },
        ZenSku { sku: "zen3-asr-0.6B",    hf_repo: "zenlm/zen-asr-0.6b",
            arch: ArchKind::Unsupported("qwen3-asr"), quant: None,
            modality: Modality::Audio },
        ZenSku { sku: "zen3-asr-aligner", hf_repo: "zenlm/zen-asr-aligner-0.6b",
            arch: ArchKind::Unsupported("qwen3-asr"), quant: None,
            modality: Modality::Audio },
        ZenSku { sku: "zen3-tts",               hf_repo: "zenlm/zen-tts-1.7b",
            arch: ArchKind::Unsupported("qwen3-tts"), quant: None,
            modality: Modality::Audio },
        ZenSku { sku: "zen3-tts-0.6B",          hf_repo: "zenlm/zen-tts-0.6b",
            arch: ArchKind::Unsupported("qwen3-tts"), quant: None,
            modality: Modality::Audio },
        ZenSku { sku: "zen3-tts-voice-design",  hf_repo: "zenlm/zen-tts-voicedesign-1.7b",
            arch: ArchKind::Unsupported("qwen3-tts"), quant: None,
            modality: Modality::Audio },
        ZenSku { sku: "zen3-tts-custom-voice",  hf_repo: "zenlm/zen-tts-customvoice-1.7b",
            arch: ArchKind::Unsupported("qwen3-tts"), quant: None,
            modality: Modality::Audio },

        // Zen3 embeddings (text-only, in-tree). zen3-embedding-medium /
        // zen3-embedding-small ghost aliases were sunset 2026-05-30; use the
        // zen5-embedding-{0.6B,4B,8B} ladder for new integrations.
        ZenSku { sku: "zen3-embedding",        hf_repo: "zenlm/zen-embedding",
            arch: ArchKind::Supported("qwen3"), quant: None,
            modality: Modality::Embedding },
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
            "zen5", "zen5-pro", "zen5-max", "zen5-flash", "zen5-coder",
            "zen5-embedding-8B", "zen3-vl", "zen3-asr", "zen3-nano",
        ] {
            assert!(lookup(sku).is_some(), "missing SKU `{sku}`");
        }
    }

    #[test]
    fn sunset_skus_are_absent() {
        // Zen4 generation + virtual zen3 aliases were removed 2026-05-30.
        for sku in [
            "zen4", "zen4-pro", "zen4-max", "zen4.1", "zen4-mini", "zen4-ultra",
            "zen4-thinking", "zen4-coder", "zen4-coder-flash", "zen4-coder-pro",
            "zen5-ultra",
            "zen3-vl-reranker-2B", "zen3-vl-reranker-8B",
            "zen3-vl-embedding-2B", "zen3-vl-embedding-8B",
            "zen3-web-8B", "zen3-web-14B", "zen3-web-32B",
            "zen3-embedding-small", "zen3-embedding-medium",
        ] {
            assert!(lookup(sku).is_none(), "sunset SKU `{sku}` still in registry");
        }
    }

    #[test]
    fn unsupported_arches_are_explicit() {
        let pro = lookup("zen5-pro").unwrap();
        assert!(!pro.arch.is_supported());
        assert_eq!(pro.arch.name(), "deepseek-v4-flash");
        let vl = lookup("zen3-vl").unwrap();
        assert!(!vl.arch.is_supported());
        assert_eq!(vl.arch.name(), "qwen3-vl");
    }

    #[test]
    fn supported_arch_names_match_normal_loader() {
        for e in all().iter().filter(|e| e.arch.is_supported()) {
            match e.arch.name() {
                "qwen3" | "qwen3moe" | "deepseekv3" | "glm4moe" | "llama"
                | "mixtral" | "gpt_oss" => {}
                other => panic!("SKU `{}` uses arch `{}` not in NormalLoaderType",
                    e.sku, other),
            }
        }
    }
}
