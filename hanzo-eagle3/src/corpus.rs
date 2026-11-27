//! Grounded repository corpus loader, SHA-256 manifest verification, and deterministic
//! workload suite builder for mixed prose/code, repetitive code edits, high-entropy prose,
//! and the 20-turn coding-agent loop.

use std::fs;
use std::path::{Path, PathBuf};

use crate::rng::sha256_hex;

/// Compact deterministic ASCII-128 vocabulary size.
pub const VOCAB_SIZE: usize = 128;

/// Relative paths of the 3 versioned repository corpus files.
pub const CORPUS_FILES: [&str; 3] = [
    "calibration_data/calibration_datav3_small.txt",
    "chat_templates/llama3.json",
    "orderings/xlora-paper-ordering.json",
];

#[derive(Clone, Debug)]
pub struct CorpusFileRecord {
    pub rel_path: String,
    pub bytes: usize,
    pub sha256: String,
    pub tokens: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct GroundedCorpus {
    pub files: Vec<CorpusFileRecord>,
    pub total_bytes: usize,
    pub combined_sha256: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkloadKind {
    CodeEditRepetitive,
    MixedProseCode,
    CreativeHighEntropyProse,
    AgentLoop20Turn,
}

impl WorkloadKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CodeEditRepetitive => "code_edit_repetitive",
            Self::MixedProseCode => "mixed_prose_code",
            Self::CreativeHighEntropyProse => "creative_high_entropy_prose",
            Self::AgentLoop20Turn => "20_turn_coding_agent_loop",
        }
    }
}

/// One decoding turn / request in a workload benchmark.
#[derive(Clone, Debug)]
pub struct DecodeTurn {
    pub turn_id: usize,
    pub kind_label: &'static str,
    pub prompt: Vec<u32>,
    pub max_new_tokens: usize,
    /// Optional high-entropy logit perturbation scale (0.0 for normal prose/code,
    /// > 0.0 for high-entropy creative decoding regime).
    pub entropy_jitter: f32,
}

/// Deterministically tokenize bytes into `[0, VOCAB_SIZE)`.
pub fn tokenize_bytes(data: &[u8]) -> Vec<u32> {
    data.iter()
        .map(|&b| {
            if b < VOCAB_SIZE as u8 {
                b as u32
            } else {
                (b & 0x7F) as u32
            }
        })
        .collect()
}

/// Locate repository root containing `calibration_data/calibration_datav3_small.txt`.
pub fn find_repo_root() -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.clone());
        if let Some(p) = cwd.parent() {
            candidates.push(p.to_path_buf());
        }
        if let Some(p) = cwd.parent().and_then(|p| p.parent()) {
            candidates.push(p.to_path_buf());
        }
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest_dir.clone());
    if let Some(p) = manifest_dir.parent() {
        candidates.push(p.to_path_buf());
    }

    for root in candidates {
        if root.join(CORPUS_FILES[0]).is_file()
            && root.join(CORPUS_FILES[1]).is_file()
            && root.join(CORPUS_FILES[2]).is_file()
        {
            return Ok(root);
        }
    }
    Err(
        "could not locate repository corpus files (calibration_data/, chat_templates/, orderings/)"
            .to_string(),
    )
}

impl GroundedCorpus {
    /// Load and SHA-256 hash all 3 versioned repository corpus files.
    pub fn load_from_repo(repo_root: &Path) -> Result<Self, String> {
        let mut files = Vec::with_capacity(CORPUS_FILES.len());
        let mut combined = Vec::new();
        let mut total_bytes = 0usize;

        for rel in CORPUS_FILES {
            let full = repo_root.join(rel);
            let raw =
                fs::read(&full).map_err(|e| format!("failed reading corpus file {rel}: {e}"))?;
            let hash = sha256_hex(&raw);
            let tokens = tokenize_bytes(&raw);
            total_bytes += raw.len();
            combined.extend_from_slice(&raw);
            files.push(CorpusFileRecord {
                rel_path: rel.to_string(),
                bytes: raw.len(),
                sha256: hash,
                tokens,
            });
        }

        let combined_sha256 = sha256_hex(&combined);
        Ok(Self {
            files,
            total_bytes,
            combined_sha256,
        })
    }

    /// Build deterministic training windows `(seq_len = 64)` covering prose, chat template,
    /// and structured JSON ordering slices from the grounded corpus.
    pub fn training_sequences(&self, num_seqs: usize, seq_len: usize) -> Vec<Vec<u32>> {
        let mut seqs = Vec::with_capacity(num_seqs);
        let prose = &self.files[0].tokens;
        let chat = &self.files[1].tokens;
        let json = &self.files[2].tokens;

        for i in 0..num_seqs {
            let src = match i % 5 {
                0..=2 => prose,
                3 => json,
                _ => chat,
            };
            if src.len() <= seq_len {
                let mut v = Vec::with_capacity(seq_len);
                while v.len() < seq_len {
                    let take = (seq_len - v.len()).min(src.len());
                    v.extend_from_slice(&src[..take]);
                }
                seqs.push(v);
            } else {
                let stride = ((src.len() - seq_len) / num_seqs.max(1)).max(7);
                let start = (i * stride) % (src.len() - seq_len);
                seqs.push(src[start..start + seq_len].to_vec());
            }
        }
        seqs
    }

    /// Construct the 4 required benchmark workload suites directly from the grounded corpus.
    pub fn build_workload_turns(&self, kind: WorkloadKind) -> Vec<DecodeTurn> {
        let prose = &self.files[0].tokens;
        let chat = &self.files[1].tokens;
        let json = &self.files[2].tokens;

        match kind {
            WorkloadKind::CodeEditRepetitive => {
                // High-overlap code/JSON edit diffs: prompt contains structured blocks from
                // `xlora-paper-ordering.json` and `llama3.json` followed by an edit prefix
                // that triggers repeated key/template spans.
                (0..4)
                    .map(|i| {
                        let mut prompt = Vec::with_capacity(144);
                        if i % 2 == 0 {
                            let base = (i * 96) % (json.len().saturating_sub(140).max(1));
                            let end = (base + 112).min(json.len());
                            prompt.extend_from_slice(&json[base..end]);
                            // Repeat the opening 24-token structural header so the edit continues
                            // the same schema pattern (exact code-edit diff scenario).
                            let repeat_len = 24.min(end - base);
                            prompt.extend_from_slice(&json[base..base + repeat_len]);
                        } else {
                            let take = 112.min(chat.len());
                            prompt.extend_from_slice(&chat[..take]);
                            let repeat_len = 24.min(take);
                            prompt.extend_from_slice(&chat[..repeat_len]);
                        }
                        DecodeTurn {
                            turn_id: i,
                            kind_label: "code_edit_repetitive",
                            prompt,
                            max_new_tokens: 48,
                            entropy_jitter: 0.0,
                        }
                    })
                    .collect()
            }
            WorkloadKind::MixedProseCode => {
                // Novel prose & mixed template prompts (no trivial verbatim prefix copy):
                // suffix lookup rarely hits long matches, while EAGLE-3 feature fusion shines.
                (0..4)
                    .map(|i| {
                        let start = 1024 + i * 733;
                        let s = start % prose.len().saturating_sub(64).max(1);
                        let mut prompt = prose[s..s + 40].to_vec();
                        if i % 2 == 1 {
                            // Mix in short template header without repeating the continuation.
                            prompt.extend_from_slice(&chat[..16.min(chat.len())]);
                        }
                        DecodeTurn {
                            turn_id: i,
                            kind_label: "mixed_prose_code",
                            prompt,
                            max_new_tokens: 48,
                            entropy_jitter: 0.0,
                        }
                    })
                    .collect()
            }
            WorkloadKind::CreativeHighEntropyProse => {
                // High-entropy creative / brainstorming turns where next-token uncertainty is
                // elevated, causing fixed speculative drafting to suffer low acceptance (~1.1-1.3).
                (0..4)
                    .map(|i| {
                        let start = 20_000 + i * 1997;
                        let s = start % prose.len().saturating_sub(64).max(1);
                        let prompt = prose[s..s + 40].to_vec();
                        DecodeTurn {
                            turn_id: i,
                            kind_label: "creative_high_entropy_prose",
                            prompt,
                            max_new_tokens: 48,
                            entropy_jitter: 2.8,
                        }
                    })
                    .collect()
            }
            WorkloadKind::AgentLoop20Turn => {
                // Realistic 20-turn coding-agent loop combining:
                // - 8 repetitive code-edit/tool-diff turns (PromptLookup wins)
                // - 8 mixed prose+code reasoning turns (EAGLE-3 Tree/Linear wins)
                // - 4 creative/ambiguous design turns (PlainDecode stand-down wins)
                let mut turns = Vec::with_capacity(20);
                for t in 0..20 {
                    let phase = match t {
                        0..=3 => 0,   // initial code-edit/diff phase
                        4..=7 => 1,   // prose+code reasoning phase
                        8..=11 => 2,  // high-entropy architectural brainstorming
                        12..=15 => 1, // mixed implementation reasoning
                        _ => 0,       // final code-edit/patch application turns
                    };
                    match phase {
                        0 => {
                            let base = (t * 64) % (json.len().saturating_sub(128).max(1));
                            let end = (base + 96).min(json.len());
                            let mut prompt = json[base..end].to_vec();
                            let rep = 20.min(end - base);
                            prompt.extend_from_slice(&json[base..base + rep]);
                            turns.push(DecodeTurn {
                                turn_id: t,
                                kind_label: "agent_code_edit",
                                prompt,
                                max_new_tokens: 32,
                                entropy_jitter: 0.0,
                            });
                        }
                        1 => {
                            let start = 3072 + t * 887;
                            let s = start % prose.len().saturating_sub(64).max(1);
                            let mut prompt = prose[s..s + 36].to_vec();
                            let c_off = (t * 13) % (chat.len().saturating_sub(16).max(1));
                            prompt.extend_from_slice(
                                &chat[c_off..c_off + 12.min(chat.len() - c_off)],
                            );
                            turns.push(DecodeTurn {
                                turn_id: t,
                                kind_label: "agent_mixed_prose_code",
                                prompt,
                                max_new_tokens: 32,
                                entropy_jitter: 0.0,
                            });
                        }
                        _ => {
                            let start = 40_000 + t * 1553;
                            let s = start % prose.len().saturating_sub(64).max(1);
                            let prompt = prose[s..s + 36].to_vec();
                            turns.push(DecodeTurn {
                                turn_id: t,
                                kind_label: "agent_high_entropy_design",
                                prompt,
                                max_new_tokens: 32,
                                entropy_jitter: 2.8,
                            });
                        }
                    }
                }
                turns
            }
        }
    }
}
