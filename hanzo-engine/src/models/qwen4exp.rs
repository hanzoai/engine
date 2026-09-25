#![allow(clippy::cast_possible_truncation)]

//! Qwen3.8-Flash-Next (`Qwen4ExpForConditionalGeneration`) from its Hugging Face safetensors
//! snapshot: the config and the manifest of every tensor the text model reads.
//!
//! The manifest is built from the config alone and is the loader's contract: the checkpoint's
//! headers must equal it, apart from the prefixes on [`DEFERRED`], and the loader must read exactly
//! it. Routed experts are NVFP4 (ModelOpt: packed E2M1 codes, E4M3 block scales, an F32 global
//! scale and an F32 activation scale); the dense side layers are block FP8 (E4M3 with one F32 scale
//! per 128x128 block); the rest is BF16. The n-gram table (layer 1's PLE) stays on the host.
//!
//! Source: vLLM `transformers_utils/configs/qwen4_exp.py`, `models/qwen4_exp/nvidia/model.py`.

use hanzo_ml::{DType, Result};
use serde::{Deserialize, Serialize};

/// Everything under this prefix except `lm_head.weight`.
pub(crate) const PREFIX: &str = "model.language_model.";

/// Checkpoint prefixes the text model does not read yet, and the milestone that reads them.
pub(crate) const DEFERRED: [(&str, &str); 3] = [
    ("self_attn.indexer.", "M2: QSA sparse attention"),
    ("mtp.", "M3: multi-token prediction"),
    ("model.visual.", "never: text only"),
];

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct RopeParameters {
    pub(crate) rope_theta: f64,
    pub(crate) partial_rotary_factor: f64,
    pub(crate) mrope_section: Vec<usize>,
    #[serde(default)]
    pub(crate) rope_type: Option<String>,
}

/// `text_config` of the snapshot's config.json.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct TextConfig {
    pub(crate) hidden_size: usize,
    pub(crate) num_hidden_layers: usize,
    pub(crate) layer_types: Vec<String>,
    pub(crate) hc_count: usize,
    pub(crate) hc_lowrank: usize,
    pub(crate) head_dim: usize,
    pub(crate) num_attention_heads: usize,
    pub(crate) num_key_value_heads: usize,
    pub(crate) linear_key_head_dim: usize,
    pub(crate) linear_value_head_dim: usize,
    pub(crate) linear_num_key_heads: usize,
    pub(crate) linear_num_value_heads: usize,
    pub(crate) linear_conv_kernel_dim: usize,
    pub(crate) num_experts: usize,
    pub(crate) num_experts_per_tok: usize,
    pub(crate) moe_intermediate_size: usize,
    pub(crate) shared_expert_intermediate_size: usize,
    pub(crate) vocab_size: usize,
    pub(crate) rms_norm_eps: f64,
    pub(crate) max_position_embeddings: usize,
    pub(crate) indexer_budget: usize,
    pub(crate) rope_parameters: RopeParameters,
    /// 1-based: the n-gram block runs before layer `ple_layer_ids[0] - 1`.
    pub(crate) ple_layer_ids: Vec<usize>,
    pub(crate) ngram_size: usize,
    pub(crate) heads_per_ngram: usize,
    pub(crate) ple_embed_dim: usize,
    pub(crate) ple_conv_kernel_size: usize,
    pub(crate) split_ngram_parts: usize,
    pub(crate) ngram_vocab_size_base: u64,
    pub(crate) make_ngram_vocab_size_divisible_by: u64,
    pub(crate) eos_token_id: u32,
    #[serde(default)]
    pub(crate) tie_word_embeddings: bool,
}

/// The snapshot's config.json.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Config {
    pub(crate) text_config: TextConfig,
    #[serde(default)]
    pub(crate) architectures: Vec<String>,
    #[serde(default)]
    pub(crate) quantization_config: Option<serde_json::Value>,
}

impl Config {
    pub(crate) fn text(&self) -> &TextConfig {
        &self.text_config
    }

    /// The checkpoint's quantization config (ModelOpt), which routes by the tensors present.
    pub(crate) fn quant(&self) -> Result<Option<hanzo_quant::QuantizedConfig>> {
        self.quantization_config
            .clone()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| hanzo_ml::Error::Msg(format!("quantization_config: {e}")))
    }

    /// The shared blocks' shape. The context is capped at the QSA indexer's budget, where dense
    /// attention is exact, until the sparse path lands.
    pub(crate) fn props(&self) -> crate::models::quantized_qwen3_5_moe::PropsGGUF {
        let t = &self.text_config;
        crate::models::quantized_qwen3_5_moe::PropsGGUF {
            head_count: t.num_attention_heads,
            head_count_kv: t.num_key_value_heads,
            block_count: t.num_hidden_layers,
            embedding_length: t.hidden_size,
            rms_norm_eps: t.rms_norm_eps as f32,
            max_seq_len: t.max_position_embeddings.min(t.indexer_budget),
            rope_freq_base: t.rope_parameters.rope_theta as f32,
            head_dim: t.head_dim,
            rot_dim: t.rotary_dim(),
            mrope_section: t.rope_parameters.mrope_section.clone(),
            full_attention_interval: t
                .layer_types
                .iter()
                .position(|l| l == "full_attention")
                .map_or(4, |i| i + 1),
            conv_kernel: t.linear_conv_kernel_dim,
            head_k_dim: t.linear_key_head_dim,
            head_v_dim: t.linear_value_head_dim,
            num_k_heads: t.linear_num_key_heads,
            num_v_heads: t.linear_num_value_heads,
            num_experts: Some(t.num_experts),
            num_experts_per_tok: t.num_experts_per_tok,
            moe_intermediate_size: t.moe_intermediate_size,
            is_moe: true,
            nextn_predict_layers: 0,
        }
    }
}

impl TextConfig {
    /// Whether layer `i` is gated attention (else a gated delta-net).
    pub(crate) fn attention(&self, i: usize) -> bool {
        self.layer_types[i] == "full_attention"
    }

    /// The layer whose input the n-gram delta joins.
    pub(crate) fn ple_layer(&self) -> Result<usize> {
        match self.ple_layer_ids.as_slice() {
            [id] if *id >= 1 => Ok(id - 1),
            other => hanzo_ml::bail!("qwen4exp needs exactly one n-gram layer, got {other:?}"),
        }
    }

    pub(crate) fn rotary_dim(&self) -> usize {
        (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize
    }

    /// Device bytes of one sequence's recurrent state: every gated delta-net layer's f32 state
    /// `[v_heads, dk, dv]` and bf16 conv window `[conv_dim, kernel]`, and the n-gram block's bf16
    /// conv history `[streams·hidden, (kernel - 1)·ngram_size]`.
    pub(crate) fn state_bytes_per_sequence(&self) -> usize {
        let key_dim = self.linear_num_key_heads * self.linear_key_head_dim;
        let conv_dim = 2 * key_dim + self.linear_num_value_heads * self.linear_value_head_dim;
        let gdn = self.linear_num_value_heads
            * self.linear_key_head_dim
            * self.linear_value_head_dim
            * DType::F32.size_in_bytes()
            + conv_dim * self.linear_conv_kernel_dim * DType::BF16.size_in_bytes();
        let layers = (0..self.num_hidden_layers).filter(|&i| !self.attention(i)).count();
        let ngram = self.hc_count
            * self.hidden_size
            * (self.ple_conv_kernel_size - 1)
            * self.ngram_size
            * DType::BF16.size_in_bytes();
        layers * gdn + ngram
    }

    /// The n-gram heads: (ngram_size - 1) orders of `heads_per_ngram` each.
    pub(crate) fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Per-head table sizes: the `h+1`-th prime past `ngram_vocab_size_base - 1` for head `h` of
    /// the (single) PLE layer, as vLLM's `_make_vocab_layout`.
    pub(crate) fn ngram_head_sizes(&self) -> Vec<u64> {
        let mut sizes = Vec::with_capacity(self.ngram_heads());
        let mut p = self.ngram_vocab_size_base - 1;
        for _ in 0..self.ngram_heads() {
            p += 1;
            while !is_prime(p) {
                p += 1;
            }
            sizes.push(p);
        }
        sizes
    }

    /// Rows of the n-gram table, padded to `make_ngram_vocab_size_divisible_by`, and the rows per
    /// checkpoint shard.
    pub(crate) fn ngram_rows(&self) -> (u64, u64) {
        let total: u64 = self.ngram_head_sizes().iter().sum();
        let d = self.make_ngram_vocab_size_divisible_by;
        let padded = total.div_ceil(d) * d;
        (padded, padded.div_ceil(self.split_ngram_parts as u64))
    }
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mut d = 2;
    while d * d <= n {
        if n % d == 0 {
            return false;
        }
        d += 1;
    }
    true
}

/// Where a tensor lives once loaded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Role {
    /// Uploaded to the model's device.
    Device,
    /// Read on the host: the n-gram hash buffers, the table and its scale.
    Host,
}

/// One checkpoint tensor: full name, stored dtype, shape and role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Entry {
    pub(crate) name: String,
    pub(crate) dtype: DType,
    pub(crate) shape: Vec<usize>,
    pub(crate) role: Role,
}

impl Entry {
    pub(crate) fn bytes(&self) -> usize {
        self.shape.iter().product::<usize>() * self.dtype.size_in_bytes()
    }
}

/// Every tensor the text model reads, in load order.
pub(crate) fn manifest(cfg: &TextConfig) -> Result<Vec<Entry>> {
    let mut out = Vec::new();
    let mut put = |name: String, dtype: DType, shape: &[usize], role: Role| {
        out.push(Entry {
            name,
            dtype,
            shape: shape.to_vec(),
            role,
        });
    };
    let lm = |s: &str| format!("{PREFIX}{s}");
    let h = cfg.hidden_size;
    let hc = cfg.hc_count * h;
    let rank = cfg.hc_lowrank;
    let blocks = |n: usize| n.div_ceil(128);
    let fp8 = |put: &mut dyn FnMut(String, DType, &[usize], Role), p: String, n: usize, k: usize| {
        put(format!("{p}.weight"), DType::F8E4M3, &[n, k], Role::Device);
        put(
            format!("{p}.weight_scale_inv"),
            DType::F32,
            &[blocks(n), blocks(k)],
            Role::Device,
        );
    };

    put(
        lm("embed_tokens.weight"),
        DType::BF16,
        &[cfg.vocab_size, h],
        Role::Device,
    );
    let mixer = |put: &mut dyn FnMut(String, DType, &[usize], Role), p: String, inject: bool| {
        put(format!("{p}.hc_norm.weight"), DType::BF16, &[hc], Role::Device);
        put(
            format!("{p}.input_mix_weight_down.weight"),
            DType::BF16,
            &[rank, hc],
            Role::Device,
        );
        put(
            format!("{p}.input_mix_weight_up.weight"),
            DType::BF16,
            &[hc, rank],
            Role::Device,
        );
        if inject {
            put(
                format!("{p}.block_inject_weight.weight"),
                DType::BF16,
                &[cfg.hc_count, hc],
                Role::Device,
            );
        }
    };

    let ple_layer = cfg.ple_layer()?;
    let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let q_dim = cfg.num_attention_heads * cfg.head_dim;
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    let ff = cfg.moe_intermediate_size;
    for i in 0..cfg.num_hidden_layers {
        let l = |s: &str| lm(&format!("layers.{i}.{s}"));
        if i == ple_layer {
            let p = |s: &str| l(&format!("ple.{s}"));
            put(p("key_proj.weight"), DType::BF16, &[hc, cfg.ple_embed_dim], Role::Device);
            put(p("value_proj.weight"), DType::BF16, &[h, cfg.ple_embed_dim], Role::Device);
            for n in ["norm_key", "norm_query", "norm_conv"] {
                put(p(&format!("{n}.weight")), DType::BF16, &[hc], Role::Device);
            }
            put(
                p("conv1d.weight"),
                DType::BF16,
                &[hc, 1, cfg.ple_conv_kernel_size],
                Role::Device,
            );
            let e = |s: &str| p(&format!("ple_embedding.{s}"));
            put(e("layer_multipliers"), DType::I64, &[cfg.ngram_size], Role::Host);
            for n in ["ngram_heads_vocab_sizes", "ngram_heads_offsets"] {
                put(e(n), DType::I64, &[cfg.ngram_heads()], Role::Host);
            }
            let (_, rows) = cfg.ngram_rows();
            let width = cfg.ple_embed_dim / cfg.ngram_heads();
            for s in 0..cfg.split_ngram_parts {
                put(
                    e(&format!("ngram_embedding.shard_{s}.weight")),
                    DType::F8E4M3,
                    &[rows as usize, width],
                    Role::Host,
                );
            }
            put(e("ngram_embedding.weight_scale"), DType::BF16, &[1], Role::Host);
        }
        mixer(&mut put, l("attn_hyper_connection"), true);
        mixer(&mut put, l("mlp_hyper_connection"), true);
        if cfg.attention(i) {
            let a = |s: &str| l(&format!("self_attn.{s}"));
            // q_proj holds [q | gate] per head.
            fp8(&mut put, a("q_proj"), 2 * q_dim, h);
            fp8(&mut put, a("k_proj"), kv_dim, h);
            fp8(&mut put, a("v_proj"), kv_dim, h);
            fp8(&mut put, a("o_proj"), h, q_dim);
            put(a("q_norm.weight"), DType::BF16, &[cfg.head_dim], Role::Device);
            put(a("k_norm.weight"), DType::BF16, &[cfg.head_dim], Role::Device);
        } else {
            let g = |s: &str| l(&format!("linear_attn.{s}"));
            let v = cfg.linear_num_value_heads;
            fp8(&mut put, g("in_proj_qkv"), 2 * key_dim + value_dim, h);
            fp8(&mut put, g("in_proj_z"), value_dim, h);
            fp8(&mut put, g("out_proj"), h, value_dim);
            put(g("in_proj_a.weight"), DType::BF16, &[v, h], Role::Device);
            put(g("in_proj_b.weight"), DType::BF16, &[v, h], Role::Device);
            put(
                g("conv1d.weight"),
                DType::BF16,
                &[2 * key_dim + value_dim, 1, cfg.linear_conv_kernel_dim],
                Role::Device,
            );
            put(g("A_log"), DType::BF16, &[v], Role::Device);
            put(g("dt_bias"), DType::BF16, &[v], Role::Device);
            put(
                g("norm.weight"),
                DType::BF16,
                &[cfg.linear_value_head_dim],
                Role::Device,
            );
        }
        let m = |s: &str| l(&format!("mlp.{s}"));
        put(m("gate.weight"), DType::BF16, &[cfg.num_experts, h], Role::Device);
        for e in 0..cfg.num_experts {
            for (proj, n, k) in [("gate_proj", ff, h), ("up_proj", ff, h), ("down_proj", h, ff)] {
                let p = |s: &str| m(&format!("experts.{e}.{proj}.{s}"));
                put(p("weight"), DType::U8, &[n, k / 2], Role::Device);
                put(p("weight_scale"), DType::F8E4M3, &[n, k / 16], Role::Device);
                put(p("weight_scale_2"), DType::F32, &[], Role::Device);
                put(p("input_scale"), DType::F32, &[], Role::Device);
            }
        }
        let sff = cfg.shared_expert_intermediate_size;
        fp8(&mut put, m("shared_expert.gate_proj"), sff, h);
        fp8(&mut put, m("shared_expert.up_proj"), sff, h);
        fp8(&mut put, m("shared_expert.down_proj"), h, sff);
        put(m("shared_expert_gate.weight"), DType::BF16, &[1, h], Role::Device);
    }
    mixer(&mut put, lm("hyper_connection_mixer"), false);
    put(
        "lm_head.weight".to_string(),
        DType::BF16,
        &[cfg.vocab_size, h],
        Role::Device,
    );
    Ok(out)
}

/// The deferral a checkpoint name falls under, if any. `self_attn.indexer.` matches inside the
/// text model's layers, the others at the start of the name.
pub(crate) fn deferred(name: &str) -> Option<&'static str> {
    DEFERRED.iter().find_map(|(p, _)| {
        let hit = if p.starts_with("self_attn.") {
            name.starts_with(PREFIX) && name.contains(&format!(".{p}"))
        } else {
            name.starts_with(p)
        };
        hit.then_some(*p)
    })
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::{BTreeMap, HashMap, HashSet};
    use std::path::{Path, PathBuf};

    use super::*;

    /// The fp8hybrid snapshot under the Hugging Face cache.
    pub(crate) const SNAPSHOT: &str = "models--nvidia--Qwen3.8-Flash-Next-NVFP4/snapshots/fc694b54fb0174e0913e6adf86691ef85a4ead47-fp8hybrid";

    pub(crate) fn snapshot() -> PathBuf {
        let hub = std::env::var_os("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
            .unwrap_or_else(|| {
                PathBuf::from(std::env::var_os("HOME").expect("HOME"))
                    .join(".cache/huggingface/hub")
            });
        hub.join(SNAPSHOT)
    }

    /// The snapshot's weight files: every `*.safetensors`, never the `.bak` originals.
    pub(crate) fn weights(dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("snapshot dir")
            .map(|e| e.expect("dir entry").path())
            .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
            .collect();
        files.sort();
        files
    }

    pub(crate) fn config(dir: &Path) -> Config {
        serde_json::from_slice(&std::fs::read(dir.join("config.json")).expect("config.json"))
            .expect("config parses")
    }

    /// One header entry: dtype string, shape, file and absolute byte range.
    #[derive(Debug, Clone)]
    pub(crate) struct Header {
        pub(crate) dtype: String,
        pub(crate) shape: Vec<usize>,
        pub(crate) file: PathBuf,
        pub(crate) start: u64,
        pub(crate) end: u64,
    }

    /// Every tensor's header across `files`, without reading any data.
    pub(crate) fn headers(files: &[PathBuf]) -> HashMap<String, Header> {
        use std::io::Read;
        let mut out = HashMap::new();
        for f in files {
            let mut fh = std::fs::File::open(f).expect("open");
            let mut len = [0u8; 8];
            fh.read_exact(&mut len).expect("header length");
            let n = u64::from_le_bytes(len);
            let mut buf = vec![0u8; n as usize];
            fh.read_exact(&mut buf).expect("header");
            let v: BTreeMap<String, serde_json::Value> =
                serde_json::from_slice(&buf).expect("header json");
            for (name, meta) in v {
                if name == "__metadata__" {
                    continue;
                }
                let offs = meta["data_offsets"].as_array().expect("offsets");
                let base = 8 + n;
                out.insert(
                    name,
                    Header {
                        dtype: meta["dtype"].as_str().expect("dtype").to_string(),
                        shape: meta["shape"]
                            .as_array()
                            .expect("shape")
                            .iter()
                            .map(|d| d.as_u64().expect("dim") as usize)
                            .collect(),
                        file: f.clone(),
                        start: base + offs[0].as_u64().expect("start"),
                        end: base + offs[1].as_u64().expect("end"),
                    },
                );
            }
        }
        out
    }

    pub(crate) fn st_dtype(d: DType) -> &'static str {
        match d {
            DType::BF16 => "BF16",
            DType::F32 => "F32",
            DType::F8E4M3 => "F8_E4M3",
            DType::U8 => "U8",
            DType::I64 => "I64",
            other => panic!("no safetensors name for {other:?}"),
        }
    }

    /// The checkpoint's headers are the manifest plus the deferred prefixes, entry for entry.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot's headers"]
    fn checkpoint_matches_manifest() -> Result<()> {
        let dir = snapshot();
        let cfg = config(&dir);
        let text = cfg.text();
        let heads = headers(&weights(&dir));
        let manifest = manifest(text)?;

        let mut deferred_counts: HashMap<&str, usize> = HashMap::new();
        let names: HashSet<&str> = manifest.iter().map(|e| e.name.as_str()).collect();
        let mut stray = Vec::new();
        for name in heads.keys() {
            match deferred(name) {
                Some(p) => *deferred_counts.entry(p).or_default() += 1,
                None if names.contains(name.as_str()) => {}
                None => stray.push(name.clone()),
            }
        }
        stray.sort();
        assert!(stray.is_empty(), "{} tensors outside the manifest: {:?}", stray.len(), &stray[..stray.len().min(20)]);
        assert_eq!(deferred_counts.get("self_attn.indexer."), Some(&36));
        assert_eq!(deferred_counts.get("mtp."), Some(&3101));
        assert_eq!(deferred_counts.get("model.visual."), Some(&333));

        let mut wrong = Vec::new();
        for e in &manifest {
            match heads.get(&e.name) {
                None => wrong.push(format!("{} missing", e.name)),
                Some(h) if h.dtype != st_dtype(e.dtype) || h.shape != e.shape => wrong.push(format!(
                    "{}: header {} {:?}, manifest {:?} {:?}",
                    e.name, h.dtype, h.shape, e.dtype, e.shape
                )),
                Some(h) => assert_eq!((h.end - h.start) as usize, e.bytes(), "{}", e.name),
            }
        }
        assert!(wrong.is_empty(), "{:?}", &wrong[..wrong.len().min(20)]);

        let device: Vec<&Entry> = manifest.iter().filter(|e| e.role == Role::Device).collect();
        let host: Vec<&Entry> = manifest.iter().filter(|e| e.role == Role::Host).collect();
        assert_eq!(manifest.len(), 296_375);
        assert_eq!(device.len(), 296_243);
        assert_eq!(host.len(), 132);
        let device_bytes: usize = device.iter().map(|e| e.bytes()).sum();
        let host_bytes: usize = host.iter().map(|e| e.bytes()).sum();
        let table: usize = host
            .iter()
            .filter(|e| e.name.contains(".shard_"))
            .map(|e| e.bytes())
            .sum();
        assert_eq!(device_bytes, 74_895_296_000);
        assert_eq!(host_bytes, 51_200_246_042);
        assert_eq!(table, 51_200_245_760);

        // The shards are one file's contiguous run. The writer laid most of them out in numeric
        // order and a few pairs transposed, so the file order is not the shard order.
        let shard = |s: usize| {
            &heads[&format!(
                "{PREFIX}layers.{}.ple.ple_embedding.ngram_embedding.shard_{s}.weight",
                text.ple_layer().unwrap()
            )]
        };
        let mut spans: Vec<&Header> = (0..text.split_ngram_parts).map(shard).collect();
        let file = spans[0].file.clone();
        assert!(spans.iter().all(|h| h.file == file), "the table spans files");
        spans.sort_by_key(|h| h.start);
        for (a, b) in spans.iter().zip(&spans[1..]) {
            assert_eq!(a.end, b.start, "a hole sits between two shards");
        }
        assert_eq!(
            spans.last().unwrap().end - spans[0].start,
            table as u64,
            "the shards are not the whole run"
        );

        // Scale invariants over all 48 x 512 experts: gate and up share weight_scale_2, and each
        // (layer, projection) has one input_scale.
        let scalar = |name: &str| -> f32 {
            use std::io::{Read, Seek, SeekFrom};
            let h = &heads[name];
            let mut f = std::fs::File::open(&h.file).expect("open");
            f.seek(SeekFrom::Start(h.start)).expect("seek");
            let mut b = [0u8; 4];
            f.read_exact(&mut b).expect("read");
            f32::from_le_bytes(b)
        };
        let mut mismatched = Vec::new();
        let mut input_scales: BTreeMap<(usize, &str), HashSet<u32>> = BTreeMap::new();
        for l in 0..text.num_hidden_layers {
            for e in 0..text.num_experts {
                let p = |proj: &str, s: &str| {
                    format!("{PREFIX}layers.{l}.mlp.experts.{e}.{proj}.{s}")
                };
                let (g, u) = (
                    scalar(&p("gate_proj", "weight_scale_2")),
                    scalar(&p("up_proj", "weight_scale_2")),
                );
                if g.to_bits() != u.to_bits() {
                    mismatched.push(format!("layer {l} expert {e}: gate {g:e} up {u:e}"));
                }
                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    input_scales
                        .entry((l, proj))
                        .or_default()
                        .insert(scalar(&p(proj, "input_scale")).to_bits());
                }
            }
        }
        let nonuniform: Vec<String> = input_scales
            .iter()
            .filter(|(_, v)| v.len() != 1)
            .map(|((l, p), v)| format!("layer {l} {p}: {} input scales", v.len()))
            .collect();
        println!("gate/up weight_scale_2 mismatches: {}", mismatched.len());
        for m in mismatched.iter().take(20) {
            println!("  {m}");
        }
        println!("non-uniform input_scale: {}", nonuniform.len());
        for m in nonuniform.iter().take(20) {
            println!("  {m}");
        }
        assert!(mismatched.is_empty(), "{} experts differ", mismatched.len());
        assert!(nonuniform.is_empty(), "{nonuniform:?}");
        Ok(())
    }

    /// A tiny config: 4 layers (attention at 3), PLE at layer 1, small everything.
    pub(crate) fn tiny() -> TextConfig {
        serde_json::from_value(serde_json::json!({
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "hc_count": 4,
            "hc_lowrank": 32,
            "head_dim": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 6,
            "linear_conv_kernel_dim": 4,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 256,
            "shared_expert_intermediate_size": 256,
            "vocab_size": 512,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 4096,
            "indexer_budget": 2048,
            "rope_parameters": {
                "rope_theta": 10000000.0,
                "partial_rotary_factor": 0.25,
                "mrope_section": [6, 5, 5],
                "rope_type": "default"
            },
            "ple_layer_ids": [2],
            "ngram_size": 3,
            "heads_per_ngram": 2,
            "ple_embed_dim": 256,
            "ple_conv_kernel_size": 4,
            "split_ngram_parts": 4,
            "ngram_vocab_size_base": 13,
            "make_ngram_vocab_size_divisible_by": 16,
            "eos_token_id": 9
        }))
        .expect("tiny config")
    }

    /// The tiny config as a whole config.json, ModelOpt quantization included.
    pub(crate) fn tiny_config() -> Config {
        Config {
            text_config: tiny(),
            architectures: vec!["Qwen4ExpForConditionalGeneration".into()],
            quantization_config: Some(serde_json::json!({
                "quant_method": "modelopt",
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {}
            })),
        }
    }

    /// A tiny Qwen3.8-Flash-Next checkpoint: every `manifest(tiny)` entry in its real dtype, in
    /// two files; the n-gram table sits in the second behind a 2-byte scale, so its data starts
    /// at an odd 2-byte offset. Values are seeded and sized so activations stay O(1).
    pub(crate) fn tiny_checkpoint(dir: &Path) -> Result<Vec<PathBuf>> {
        use hanzo_ml::{Device, Tensor};
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let cfg = tiny();
        let mut rng = StdRng::seed_from_u64(0x7169_6e79);
        let mut uniform = |n: usize, lo: f32, hi: f32| -> Vec<f32> { (0..n).map(|_| rng.random_range(lo..hi)).collect() };
        let dev = Device::Cpu;
        let mut main: HashMap<String, Tensor> = HashMap::new();
        let mut table: Vec<(String, Vec<u8>, Vec<usize>, &str)> = vec![];
        for e in manifest(&cfg)? {
            let n: usize = e.shape.iter().product();
            let name = e.name.as_str();
            let fan_in = *e.shape.last().unwrap_or(&1) as f32;
            if name.contains(".shard_") {
                let bytes: Vec<u8> = uniform(n, -0.5, 0.5)
                    .into_iter()
                    .map(|v| float8::F8E4M3::from_f32(v * 8.0).to_bits())
                    .collect();
                table.push((e.name.clone(), bytes, e.shape.clone(), "F8_E4M3"));
                continue;
            }
            if name.ends_with("ngram_embedding.weight_scale") {
                table.insert(0, (e.name.clone(), half::bf16::from_f32(0.05).to_le_bytes().to_vec(), e.shape.clone(), "BF16"));
                continue;
            }
            let t = match e.dtype {
                DType::I64 => {
                    let v: Vec<i64> = if name.ends_with("layer_multipliers") {
                        vec![3, 5, 7]
                    } else if name.ends_with("vocab_sizes") {
                        vec![13, 17, 19, 23]
                    } else {
                        vec![0, 13, 30, 49]
                    };
                    Tensor::from_vec(v, n, &dev)?
                }
                DType::U8 => Tensor::from_vec(
                    uniform(n, 0.0, 255.99).into_iter().map(|v| v as u8).collect::<Vec<u8>>(),
                    e.shape.as_slice(),
                    &dev,
                )?,
                DType::F8E4M3 if name.ends_with("weight_scale") => {
                    Tensor::from_vec(uniform(n, 0.5, 1.5), e.shape.as_slice(), &dev)?.to_dtype(DType::F8E4M3)?
                }
                DType::F8E4M3 => Tensor::from_vec(uniform(n, -64.0, 64.0), e.shape.as_slice(), &dev)?.to_dtype(DType::F8E4M3)?,
                DType::F32 if name.ends_with("weight_scale_inv") => {
                    // the matching weight's fan-in sets the block scale: code 64 ~ 1/sqrt(k)
                    let k = if name.contains("out_proj") || name.contains("o_proj") || name.contains("down_proj") {
                        e.shape[1] as f32 * 128.0
                    } else {
                        cfg.hidden_size as f32
                    };
                    Tensor::from_vec(uniform(n, 0.8, 1.2).into_iter().map(|v| v / (64.0 * k.sqrt())).collect(), e.shape.as_slice(), &dev)?
                }
                DType::F32 if name.ends_with("weight_scale_2") => {
                    // gate and up share it; a function of the expert and projection family
                    let x: usize = name.split('.').find_map(|p| p.parse().ok()).unwrap_or(0);
                    Tensor::new(0.01f32 * (1.0 + (x % 3) as f32 * 0.1), &dev)?
                }
                DType::F32 => Tensor::new(0.03f32, &dev)?,
                DType::BF16 => {
                    let (lo, hi) = if name.ends_with("A_log") {
                        (-1.0, 0.5)
                    } else if name.ends_with("dt_bias") || name.contains("conv1d") {
                        (-0.5, 0.5)
                    } else if name.ends_with("linear_attn.norm.weight") {
                        (0.7, 1.3)
                    } else if e.shape.len() == 1 {
                        (-0.1, 0.1)
                    } else {
                        let a = 1.0 / fan_in.sqrt();
                        (-a, a)
                    };
                    Tensor::from_vec(uniform(n, lo, hi), e.shape.as_slice(), &dev)?.to_dtype(DType::BF16)?
                }
                d => panic!("tiny checkpoint: {d:?}"),
            };
            main.insert(e.name.clone(), t);
        }
        let a = dir.join("model-00001-of-00002.safetensors");
        hanzo_ml::safetensors::save(&main, &a)?;
        let mut header = serde_json::Map::new();
        let mut at = 0usize;
        let mut data = vec![];
        for (name, bytes, shape, dtype) in &table {
            header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [at, at + bytes.len()]}));
            at += bytes.len();
            data.extend(bytes);
        }
        let json = serde_json::to_vec(&header).map_err(hanzo_ml::Error::wrap)?;
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend(&json);
        file.extend(&data);
        let b = dir.join("model-00002-of-00002.safetensors");
        std::fs::write(&b, &file).map_err(hanzo_ml::Error::wrap)?;
        Ok(vec![a, b])
    }

    #[test]
    fn manifest_is_closed_under_config() -> Result<()> {
        let m = manifest(&tiny())?;
        let mut seen = HashSet::new();
        for e in &m {
            assert!(seen.insert(e.name.as_str()), "duplicate {}", e.name);
            assert!(deferred(&e.name).is_none(), "{} is deferred", e.name);
        }
        assert!(m.iter().any(|e| e.role == Role::Host));
        assert!(m.iter().any(|e| e.role == Role::Device));
        for d in [DType::BF16, DType::F32, DType::F8E4M3, DType::U8, DType::I64] {
            assert!(m.iter().any(|e| e.dtype == d), "no {d:?} entry");
        }
        // 4 heads over primes past 12: 13, 17, 19, 23 -> 72 rows, padded to 80, 20 per shard.
        assert_eq!(tiny().ngram_head_sizes(), [13, 17, 19, 23]);
        assert_eq!(tiny().ngram_rows(), (80, 20));
        Ok(())
    }

    // ---------------------------------------------------------------------------------------
    // Real-checkpoint test support
    // ---------------------------------------------------------------------------------------

    /// A varbuilder over the snapshot's weight files (every `*.safetensors`, never `.bak`).
    pub(crate) fn vb(dev: &hanzo_ml::Device) -> Result<hanzo_quant::ShardedVarBuilder> {
        let files = weights(&snapshot());
        unsafe {
            hanzo_quant::ShardedSafeTensors::sharded(
                &files,
                DType::BF16,
                dev,
                None,
                std::sync::Arc::new(|_| true),
            )
        }
    }

    /// A backend that logs the full name of every tensor read through it.
    pub(crate) struct Recorder {
        inner: hanzo_quant::safetensors::MmapedSafetensors,
        pub(crate) names: std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<String>>>,
    }

    impl Recorder {
        pub(crate) fn over(
            files: &[PathBuf],
            dtype: DType,
            dev: &hanzo_ml::Device,
        ) -> Result<(
            hanzo_quant::ShardedVarBuilder,
            std::sync::Arc<std::sync::Mutex<std::collections::BTreeSet<String>>>,
        )> {
            let names = std::sync::Arc::new(std::sync::Mutex::new(Default::default()));
            let inner = unsafe { hanzo_quant::safetensors::MmapedSafetensors::multi(files)? };
            let rec = Recorder {
                inner,
                names: names.clone(),
            };
            Ok((
                hanzo_quant::ShardedSafeTensors::wrap(Box::new(rec), dtype, dev.clone()),
                names,
            ))
        }
    }

    impl hanzo_nn::var_builder::SimpleBackend for Recorder {
        fn get(
            &self,
            s: hanzo_ml::Shape,
            name: &str,
            h: hanzo_nn::Init,
            dtype: DType,
            dev: &hanzo_ml::Device,
        ) -> Result<hanzo_ml::Tensor> {
            self.names.lock().unwrap().insert(name.to_string());
            hanzo_nn::var_builder::SimpleBackend::get(&self.inner, s, name, h, dtype, dev)
        }
        fn get_unchecked(
            &self,
            name: &str,
            dtype: DType,
            dev: &hanzo_ml::Device,
        ) -> Result<hanzo_ml::Tensor> {
            self.names.lock().unwrap().insert(name.to_string());
            hanzo_nn::var_builder::SimpleBackend::get_unchecked(&self.inner, name, dtype, dev)
        }
        fn contains_tensor(&self, name: &str) -> bool {
            hanzo_nn::var_builder::SimpleBackend::contains_tensor(&self.inner, name)
        }
    }

    fn load_fixture(file: &str) -> HashMap<String, hanzo_ml::Tensor> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(file);
        hanzo_ml::safetensors::load(&path, &hanzo_ml::Device::Cpu).expect("fixture loads")
    }

    /// A tensor of the CPU golden (`scripts/qwen4exp_golden.py`), on the host.
    pub(crate) fn fixture(name: &str) -> hanzo_ml::Tensor {
        static FIX: std::sync::OnceLock<HashMap<String, hanzo_ml::Tensor>> =
            std::sync::OnceLock::new();
        FIX.get_or_init(|| load_fixture("qwen4exp.safetensors"))
            .get(name)
            .unwrap_or_else(|| panic!("no golden tensor {name}"))
            .clone()
    }

    /// A tensor the served vLLM kernels produced (`scripts/qwen4exp_vectors.py`), on the host.
    pub(crate) fn vectors(name: &str) -> hanzo_ml::Tensor {
        static VEC: std::sync::OnceLock<HashMap<String, hanzo_ml::Tensor>> =
            std::sync::OnceLock::new();
        VEC.get_or_init(|| load_fixture("qwen4exp_vectors.safetensors"))
            .get(name)
            .unwrap_or_else(|| panic!("no served vector {name}"))
            .clone()
    }

    /// A tolerance: `share` of the elements within `ulps`, and every row's cosine to the
    /// reference at least `cos`. One ulp is the bf16 spacing at |reference| plus a floor of
    /// 2^-10 · rms(reference row).
    #[derive(Debug, Clone, Copy)]
    pub(crate) struct Tol {
        pub(crate) ulps: f64,
        pub(crate) share: f64,
        pub(crate) cos: f64,
    }

    impl Tol {
        pub(crate) const fn ulps(ulps: f64) -> Self {
            Self { ulps, share: 1.0, cos: 0.0 }
        }
        pub(crate) const fn most(ulps: f64, share: f64, cos: f64) -> Self {
            Self { ulps, share, cos }
        }
    }

    /// Per-element ulp distances and per-row cosines of `got` against `want`, rows along the
    /// last dim; prints the stage's max, 99.9th-percentile ulp and minimum cosine.
    pub(crate) fn assert_close(what: &str, got: &hanzo_ml::Tensor, want: &hanzo_ml::Tensor, tol: Tol) {
        let cols = *want.dims().last().expect("rank >= 1");
        let f = |t: &hanzo_ml::Tensor| -> Vec<f32> {
            t.to_dtype(DType::F32)
                .and_then(|t| t.to_device(&hanzo_ml::Device::Cpu))
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .expect("to host")
        };
        let (g, w) = (f(got), f(want));
        assert_eq!(g.len(), w.len(), "{what}: {:?} vs {:?}", got.dims(), want.dims());
        let mut ulps = Vec::with_capacity(g.len());
        let mut min_cos = 1f64;
        for (gr, wr) in g.chunks(cols).zip(w.chunks(cols)) {
            let rms = (wr.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>() / cols as f64).sqrt();
            let (mut dot, mut ng, mut nw) = (0f64, 0f64, 0f64);
            for (a, b) in gr.iter().zip(wr) {
                let (a, b) = (f64::from(*a), f64::from(*b));
                let spacing = if b == 0.0 { 0.0 } else { 2f64.powi(b.abs().log2().floor() as i32 - 7) };
                ulps.push((a - b).abs() / (spacing + rms / 1024.0).max(f64::MIN_POSITIVE));
                dot += a * b;
                ng += a * a;
                nw += b * b;
            }
            if ng > 0.0 && nw > 0.0 {
                min_cos = min_cos.min(dot / (ng.sqrt() * nw.sqrt()));
            }
        }
        let mut sorted = ulps.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let at = |q: f64| sorted[((sorted.len() as f64 * q).ceil() as usize).clamp(1, sorted.len()) - 1];
        let (max, p999) = (at(1.0), at(0.999));
        println!("{what}: max {max:.3} ulp, p99.9 {p999:.3} ulp, min row cos {min_cos:.7}");
        let q = at(tol.share);
        assert!(
            q <= tol.ulps,
            "{what}: {:.4} of elements must be within {} ulp, the {} quantile is {q:.3}",
            tol.share,
            tol.ulps,
            tol.share
        );
        assert!(min_cos >= tol.cos, "{what}: min row cosine {min_cos} < {}", tol.cos);
    }

    /// Serializes the real-checkpoint GPU tests and bounds each one's device memory: holds a
    /// process-wide lock, refuses to start unless the host has 6 GiB available and the device 3
    /// GiB free, and on drop asserts the default mempool's used-memory high-water stayed within
    /// `bound` (GB10 device allocations are not charged to a memory cgroup, so the test does it).
    pub(crate) struct Guard {
        _lock: std::sync::MutexGuard<'static, ()>,
        bound: u64,
        #[cfg(feature = "cuda")]
        pool: hanzo_ml::cuda_backend::cudarc::driver::sys::CUmemoryPool,
        pub(crate) dev: hanzo_ml::Device,
    }

    fn mem_available() -> u64 {
        let info = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        info.lines()
            .find_map(|l| l.strip_prefix("MemAvailable:"))
            .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
            .map(|kb| kb * 1024)
            .unwrap_or(0)
    }

    const GIB: f64 = (1u64 << 30) as f64;

    pub(crate) fn gpu(bound_gib: f64) -> Guard {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let avail = mem_available();
        assert!(avail as f64 >= 6.0 * GIB, "only {:.1} GiB MemAvailable", avail as f64 / GIB);
        let dev = hanzo_ml::Device::new_cuda(0).expect("a CUDA device");
        #[cfg(feature = "cuda")]
        {
            use hanzo_ml::cuda_backend::cudarc::driver::{result, sys};
            let hanzo_ml::Device::Cuda(cu) = &dev else { unreachable!() };
            let (free, _) = result::mem_get_info().expect("cuMemGetInfo");
            assert!(free as f64 >= 3.0 * GIB, "only {:.1} GiB free on the device", free as f64 / GIB);
            use hanzo_ml::backend::BackendDevice;
            let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
            let mut zero = 0u64;
            unsafe {
                let r = sys::cuDeviceGetDefaultMemPool(&mut pool, cu.cuda_stream().context().cu_device());
                assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "default mempool");
                cu.synchronize().expect("sync");
                let r = sys::cuMemPoolSetAttribute(
                    pool,
                    sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                    (&mut zero as *mut u64).cast(),
                );
                assert_eq!(r, sys::CUresult::CUDA_SUCCESS, "reset USED_MEM_HIGH");
            }
            return Guard { _lock: lock, bound: (bound_gib * GIB) as u64, pool, dev };
        }
        #[allow(unreachable_code)]
        Guard {
            _lock: lock,
            bound: (bound_gib * GIB) as u64,
            #[cfg(feature = "cuda")]
            pool: std::ptr::null_mut(),
            dev,
        }
    }

    impl Guard {
        /// The pool's used-memory high-water since the guard was taken.
        pub(crate) fn high_water(&self) -> u64 {
            #[cfg(feature = "cuda")]
            {
                use hanzo_ml::backend::BackendDevice;
                use hanzo_ml::cuda_backend::cudarc::driver::sys;
                if let hanzo_ml::Device::Cuda(cu) = &self.dev {
                    let _ = cu.synchronize();
                }
                let mut high = 0u64;
                unsafe {
                    sys::cuMemPoolGetAttribute(
                        self.pool,
                        sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                        (&mut high as *mut u64).cast(),
                    );
                }
                return high;
            }
            #[allow(unreachable_code)]
            0
        }
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            let high = self.high_water();
            println!("device pool high-water {:.3} GiB (bound {:.2})", high as f64 / GIB, self.bound as f64 / GIB);
            if !std::thread::panicking() {
                assert!(high <= self.bound, "device high-water {high} over the bound {}", self.bound);
            }
        }
    }

    /// Rows `range` of a 2-D checkpoint tensor, read from the file by offset (only those pages).
    pub(crate) fn rows(name: &str, range: std::ops::Range<usize>) -> hanzo_ml::Tensor {
        use std::io::{Read, Seek, SeekFrom};
        static HEADS: std::sync::OnceLock<HashMap<String, Header>> = std::sync::OnceLock::new();
        let heads = HEADS.get_or_init(|| headers(&weights(&snapshot())));
        let h = &heads[name];
        let dtype = match h.dtype.as_str() {
            "BF16" => DType::BF16,
            "F32" => DType::F32,
            "F8_E4M3" => DType::F8E4M3,
            "U8" => DType::U8,
            d => panic!("rows of {d}"),
        };
        let row: usize = h.shape[1..].iter().product::<usize>() * dtype.size_in_bytes();
        let mut f = std::fs::File::open(&h.file).expect("open");
        f.seek(SeekFrom::Start(h.start + (range.start * row) as u64)).expect("seek");
        let mut buf = vec![0u8; range.len() * row];
        f.read_exact(&mut buf).expect("read");
        let mut shape = h.shape.clone();
        shape[0] = range.len();
        hanzo_ml::Tensor::from_raw_buffer(&buf, dtype, &shape, &hanzo_ml::Device::Cpu).expect("tensor")
    }

    /// The golden's 24 token ids.
    pub(crate) fn tokens() -> Vec<u32> {
        fixture("tokens")
            .to_dtype(DType::U32)
            .and_then(|t| t.to_vec1::<u32>())
            .expect("tokens")
    }

    /// `[t, n·h]` golden streams as the engine's `[1, t, n, h]`.
    pub(crate) fn streams(name: &str, dev: &hanzo_ml::Device) -> hanzo_ml::Tensor {
        let x = fixture(name);
        let (t, nh) = x.dims2().expect("2-D");
        x.reshape((1, t, 4, nh / 4)).and_then(|x| x.to_device(dev)).expect("streams")
    }

    /// The branch's norm, block input, injection and combine against layer 0's golden, and the
    /// final mixer against the head's.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot; needs a CUDA device"]
    fn golden_hc() -> Result<()> {
        let g = gpu(0.1);
        let dev = g.dev.clone();
        let eps = config(&snapshot()).text().rms_norm_eps as f32;
        let root = vb(&dev)?;
        let lm = root.pp("model.language_model");
        let branch = crate::models::hyper::Branch::new(&lm.pp("layers.0.attn_hyper_connection"), 4, eps)?;
        let e = fixture("embed").to_device(&dev)?.unsqueeze(0)?;
        let x0 = crate::models::hyper::expand(&e, 4)?;
        let xn = branch.mixer.norm(&x0)?;
        assert_close("l0.attn.xn", &xn.reshape((24, 10240))?, &fixture("l0.attn.xn"), Tol::ulps(1.0));
        let (u, inj) = branch.mix(&x0)?;
        assert_close("l0.attn.u", &u.squeeze(0)?, &fixture("l0.attn.u"), Tol::ulps(1.0));
        assert_close("l0.attn.inj", &inj.squeeze(0)?, &fixture("l0.attn.inj"), Tol::ulps(1.0));
        let y = fixture("l0.attn.y").to_device(&dev)?.unsqueeze(0)?;
        let x = crate::models::hyper::combine(&x0, &y, &fixture("l0.attn.inj").to_device(&dev)?.unsqueeze(0)?)?;
        assert_close("l0.mid", &x.reshape((24, 10240))?, &fixture("l0.mid"), Tol::ulps(1.0));

        let head = crate::models::hyper::Mixer::new(&lm.pp("hyper_connection_mixer"), 4, eps)?;
        let x4 = streams("l4.x", &dev);
        let h = head.mix(&head.norm(&x4)?)?;
        assert_close("head.h", &h.squeeze(0)?, &fixture("head.h"), Tol::ulps(1.0));
        Ok(())
    }

    /// The 24 embedding rows bit-exact, and lm_head rows [0, 8192) over the golden h.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot; needs a CUDA device"]
    fn golden_embed_head() -> Result<()> {
        let g = gpu(0.1);
        let dev = g.dev.clone();
        let ids = tokens();
        let emb: Vec<hanzo_ml::Tensor> = ids
            .iter()
            .map(|&t| rows(&format!("{PREFIX}embed_tokens.weight"), t as usize..t as usize + 1))
            .collect();
        let e = hanzo_ml::Tensor::cat(&emb, 0)?.to_device(&dev)?;
        let want = fixture("embed");
        let bits = |t: &hanzo_ml::Tensor| -> Result<Vec<u16>> {
            Ok(t.to_device(&hanzo_ml::Device::Cpu)?.flatten_all()?.to_vec1::<half::bf16>()?.into_iter().map(|v| v.to_bits()).collect())
        };
        assert_eq!(bits(&e)?, bits(&want)?, "embedding rows");

        let w = rows("lm_head.weight", 0..8192).to_device(&dev)?;
        let head = hanzo_quant::UnquantLinear::new(hanzo_quant::QuantMethodConfig::Unquantized(
            hanzo_nn::Linear::new(w, None),
        ))?;
        use hanzo_quant::QuantMethod;
        let logits = head.forward(&fixture("head.h").to_device(&dev)?)?;
        assert_close("head.logits", &logits, &fixture("head.logits"), Tol::ulps(1.0));
        Ok(())
    }

    /// The guard serializes its holders and fails a test that allocates past its bound.
    #[test]
    #[ignore = "needs a CUDA device"]
    fn gpu_guard_bounds() {
        let over = std::panic::catch_unwind(|| {
            let g = gpu(0.01);
            let _t = hanzo_ml::Tensor::zeros((64 << 20,), DType::F32, &g.dev).expect("alloc");
            drop(g);
        });
        assert!(over.is_err(), "256 MiB under a 10 MiB bound must fail");
        let inside = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let threads: Vec<_> = (0..2)
            .map(|_| {
                let inside = inside.clone();
                std::thread::spawn(move || {
                    let _g = gpu(0.5);
                    let n = inside.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    assert_eq!(n, 0, "two holders at once");
                    std::thread::sleep(std::time::Duration::from_millis(50));
                    inside.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
                })
            })
            .collect();
        for t in threads {
            t.join().expect("no overlap");
        }
    }

    /// The golden tokens' chunks: prefill, continuation, decode.
    pub(crate) const CHUNKS: [usize; 3] = [16, 7, 1];

    /// Rows `at..at+len` of a `[t, ...]` host tensor on `dev`, with a leading batch dim.
    pub(crate) fn chunk(t: &hanzo_ml::Tensor, at: usize, len: usize, dev: &hanzo_ml::Device) -> hanzo_ml::Tensor {
        t.narrow(0, at, len)
            .and_then(|t| t.to_device(dev))
            .and_then(|t| t.unsqueeze(0))
            .expect("chunk")
    }

    fn quant_fixture(name: &str) -> hanzo_ml::Tensor {
        static Q: std::sync::OnceLock<HashMap<String, hanzo_ml::Tensor>> = std::sync::OnceLock::new();
        Q.get_or_init(|| {
            let path = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../hanzo-quant/tests/fixtures/vllm_vectors.safetensors");
            hanzo_ml::safetensors::load(&path, &hanzo_ml::Device::Cpu).expect("quant vectors")
        })
        .get(name)
        .unwrap_or_else(|| panic!("no quantizer vector {name}"))
        .clone()
    }

    fn fp8_codes(t: &hanzo_ml::Tensor) -> Vec<u8> {
        t.to_device(&hanzo_ml::Device::Cpu)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<float8::F8E4M3>())
            .expect("codes")
            .into_iter()
            .map(|v| v.to_bits())
            .collect()
    }

    /// Layer 3's gated attention in chunks [16, 7, 1] against the served normrope, attention
    /// core and o_proj, and the golden block output.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot; needs a CUDA device"]
    fn golden_attention() -> Result<()> {
        use crate::layers::Qwen3VLRotaryEmbedding;
        use crate::models::quantized_qwen3_5_moe::{text_mrope, GatedFullAttention};
        let g = gpu(1.45);
        let dev = g.dev.clone();
        let cfg = config(&snapshot());
        let props = cfg.props();
        let (vb, names) = Recorder::over(&weights(&snapshot()), DType::BF16, &dev)?;
        let rotary = std::sync::Arc::new(Qwen3VLRotaryEmbedding::new(
            props.rope_freq_base,
            props.rot_dim,
            &dev,
            props.mrope_section.clone(),
        )?);
        let attn = GatedFullAttention::new(
            &vb.pp("model.language_model.layers.3.self_attn"),
            &props,
            &cfg.quant()?,
            rotary.clone(),
            None,
            &dev,
            DType::BF16,
        )?;
        let mut kv = crate::pipeline::KvCache::new_normal(2, props.max_seq_len, 64);
        let mask = crate::attention::AttentionMask::None;
        let u = fixture("l3.attn.u");
        let (mut qs, mut ks, mut cores, mut outs, mut prods) = (vec![], vec![], vec![], vec![], vec![]);
        let mut at = 0;
        for len in CHUNKS {
            let x = chunk(&u, at, len, &dev);
            let cos_sin = text_mrope(&rotary, &dev, &[at], len, DType::BF16, None)?;
            let (q, k, v, gate) = attn.project(&x, &cos_sin)?;
            // [1, h, s, d] -> [s, h·d]
            let flat = |t: &hanzo_ml::Tensor| t.squeeze(0).and_then(|t| t.transpose(0, 1)).and_then(|t| t.contiguous()).and_then(|t| t.reshape((len, ())));
            qs.push(flat(&q)?);
            ks.push(flat(&k)?);
            let y = attn.attend(&q, &k, &v, &mask, &mut kv, None)?;
            cores.push(y.squeeze(0)?);
            let g32 = crate::models::gdn::sigmoid(&gate.to_dtype(DType::F32)?)?;
            prods.push(y.to_dtype(DType::F32)?.broadcast_mul(&g32)?.squeeze(0)?);
            outs.push(attn.output(&y, &gate, DType::BF16)?.squeeze(0)?);
            at += len;
        }
        let cat = |v: &[hanzo_ml::Tensor]| hanzo_ml::Tensor::cat(v, 0).expect("cat");
        assert_close("attn.q vs served", &cat(&qs), &vectors("attn.normrope.q"), Tol::ulps(1.0));
        assert_close("attn.k vs served", &cat(&ks), &vectors("attn.normrope.k"), Tol::ulps(1.0));
        assert_close("attn.core vs served (24 rows)", &cat(&cores), &vectors("attn.core24"), Tol::most(2.0, 0.999, 0.0));
        assert_close("attn.core row 23 vs served (1 row)", &cores[2], &vectors("attn.core1"), Tol::ulps(4.0));
        assert_close("attn.core vs golden", &cat(&cores), &fixture("l3.attn.core"), Tol::most(2.0, 0.999, 0.0));
        let (oq, _) = hanzo_quant::quantize::fp8(&cat(&prods), hanzo_quant::quantize::Fp8Mode::Linear)?;
        let (a, b) = (fp8_codes(&oq), fp8_codes(&quant_fixture("fp8.gated.q")));
        let differ = a.iter().zip(&b).filter(|(x, y)| x != y).count();
        println!("o_proj input codes: {differ} of {} differ from served", a.len());
        assert!(differ as f64 <= 1e-4 * a.len() as f64);
        let out = cat(&outs);
        assert_close("o_proj vs served w8a8", &out, &vectors("w8a8.o_proj"), Tol::most(1.0, 0.999, 0.0));
        assert_close("o_proj vs served w8a8 (max)", &out, &vectors("w8a8.o_proj"), Tol::ulps(2.0));
        assert_close("attn block vs golden", &out, &fixture("l3.attn.y"), Tol::most(4.0, 0.999, 0.9999));

        let want: std::collections::BTreeSet<String> = manifest(cfg.text())?
            .into_iter()
            .map(|e| e.name)
            .filter(|n| n.starts_with(&format!("{PREFIX}layers.3.self_attn.")))
            .collect();
        assert_eq!(*names.lock().unwrap(), want, "tensors read");
        Ok(())
    }

    /// The engine's tiled V heads back in the checkpoint's grouped order along `dim` (width
    /// `width` per head): grouped head `k·g + r` sits at tiled `r·K + k`.
    fn grouped(t: &hanzo_ml::Tensor, dim: usize, start: usize, heads: usize, kh: usize, width: usize) -> hanzo_ml::Tensor {
        let g = heads / kh;
        let mut parts = vec![t.narrow(dim, 0, start).expect("lead")];
        for j in 0..heads {
            let (k, r) = (j / g, j % g);
            parts.push(t.narrow(dim, start + (r * kh + k) * width, width).expect("head"));
        }
        let parts: Vec<_> = parts.into_iter().filter(|p| p.dim(dim).unwrap() > 0).collect();
        hanzo_ml::Tensor::cat(&parts, dim).expect("cat")
    }

    /// Tiled from grouped at load, with block-FP8 projections, equals the grouped `gdn.rs`
    /// reference over the same layers: the permutation is exactly the reordering.
    #[test]
    fn gdn_new_permutes_to_tiled() -> Result<()> {
        use crate::models::gdn::{GatedDeltaNet, GdnInProj, GdnLayerCache, RmsNormGated};
        use crate::models::quantized_qwen3_5_moe::{PropsGGUF, QGatedDeltaNet};
        use hanzo_ml::{Device, Tensor};
        use hanzo_quant::QuantMethod;
        let dev = Device::Cpu;
        let (kh, vh, d, h, conv) = (2usize, 6usize, 128usize, 256usize, 4usize);
        let (key_dim, value_dim) = (kh * d, vh * d);
        let conv_dim = 2 * key_dim + value_dim;
        let fp8 = |n: usize, k: usize, seed: u64| -> Result<(Tensor, Tensor)> {
            let w = (Tensor::rand(-1f32, 1f32, (n, k), &dev)? * (100.0 + seed as f64))?
                .to_dtype(DType::F8E4M3)?;
            let s = Tensor::rand(1e-3f32, 3e-3f32, (n / 128, k / 128), &dev)?;
            Ok((w, s))
        };
        let (qkv, qkv_s) = fp8(conv_dim, h, 1)?;
        let (z, z_s) = fp8(value_dim, h, 2)?;
        let (o, o_s) = fp8(h, value_dim, 3)?;
        let small = |shape: &[usize], lo: f32, hi: f32| Tensor::rand(lo, hi, shape, &dev);
        let (b, a) = (small(&[vh, h], -0.05, 0.05)?, small(&[vh, h], -0.05, 0.05)?);
        let conv1d = small(&[conv_dim, 1, conv], -0.5, 0.5)?;
        let a_log = small(&[vh], -1.0, 0.5)?;
        let dt = small(&[vh], -0.5, 0.5)?;
        let norm = small(&[d], 0.7, 1.3)?;
        let st: HashMap<String, Tensor> = [
            ("in_proj_qkv.weight", qkv.clone()),
            ("in_proj_qkv.weight_scale_inv", qkv_s.clone()),
            ("in_proj_z.weight", z.clone()),
            ("in_proj_z.weight_scale_inv", z_s.clone()),
            ("out_proj.weight", o.clone()),
            ("out_proj.weight_scale_inv", o_s.clone()),
            ("in_proj_a.weight", a.clone()),
            ("in_proj_b.weight", b.clone()),
            ("conv1d.weight", conv1d.clone()),
            ("A_log", a_log.clone()),
            ("dt_bias", dt.clone()),
            ("norm.weight", norm.clone()),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let path = std::env::temp_dir().join(format!("hanzo-gdn-tiled-{}.safetensors", std::process::id()));
        hanzo_ml::safetensors::save(&st, &path)?;
        let vb = unsafe { hanzo_quant::ShardedSafeTensors::sharded(&[&path], DType::F32, &dev, None, std::sync::Arc::new(|_| true))? };
        let props = PropsGGUF {
            head_count: 1,
            head_count_kv: 1,
            block_count: 1,
            embedding_length: h,
            rms_norm_eps: 1e-6,
            max_seq_len: 64,
            rope_freq_base: 1e4,
            head_dim: d,
            rot_dim: 32,
            mrope_section: vec![6, 5, 5],
            full_attention_interval: 4,
            conv_kernel: conv,
            head_k_dim: d,
            head_v_dim: d,
            num_k_heads: kh,
            num_v_heads: vh,
            num_experts: None,
            num_experts_per_tok: 1,
            moe_intermediate_size: 1,
            is_moe: false,
            nextn_predict_layers: 0,
        };
        let ours = QGatedDeltaNet::new(&vb, &props, &None)?;
        let block = |w: &Tensor, s: &Tensor| hanzo_quant::blockwise_fp8_moe(w.clone(), s.clone(), vec![128, 128], DType::F32);
        let dense = |w: &Tensor| -> Result<std::sync::Arc<dyn hanzo_quant::QuantMethod>> {
            Ok(std::sync::Arc::new(hanzo_quant::UnquantLinear::new(hanzo_quant::QuantMethodConfig::Unquantized(
                hanzo_nn::Linear::new(w.clone(), None),
            ))?))
        };
        let twin = GatedDeltaNet {
            in_proj: GdnInProj::Split { qkv: block(&qkv, &qkv_s)?, z: block(&z, &z_s)?, b: dense(&b)?, a: dense(&a)? },
            conv1d_weight: conv1d,
            dt_bias: dt,
            a_log,
            norm: RmsNormGated::from_weight(norm, 1e-6).sigmoid(),
            out_proj: block(&o, &o_s)?,
            num_k_heads: kh,
            num_v_heads: vh,
            head_k_dim: d,
            head_v_dim: d,
            conv_kernel_size: conv,
            key_dim,
            value_dim,
        };
        let fresh = || -> Result<GdnLayerCache> {
            Ok(GdnLayerCache {
                conv_state: Tensor::zeros((1, conv_dim, conv), DType::F32, &dev)?,
                recurrent_state: Tensor::zeros((1, vh, d, d), DType::F32, &dev)?,
                seqlen_offset: 0,
                trail: None,
            })
        };
        let (mut a_cache, mut b_cache) = (fresh()?, fresh()?);
        for len in [5usize, 1] {
            let x = Tensor::rand(-1f32, 1f32, (1, len, h), &dev)?;
            let got = ours.forward(&x, &mut a_cache)?;
            let want = twin.forward(&x, &mut b_cache)?;
            let err = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(err <= 1e-5 * scale.max(1.0), "len {len}: {err:e} off {scale:e}");
        }
        Ok(())
    }

    /// Layer 0's gated delta-net in chunks [16, 7, 1] against the served FLA prefill chain and
    /// fused decode, and the golden block output and final state.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot; needs a CUDA device"]
    fn golden_gdn() -> Result<()> {
        use crate::models::gdn::GdnLayerCache;
        use crate::models::quantized_qwen3_5_moe::QGatedDeltaNet;
        use hanzo_ml::Tensor;
        let g = gpu(1.45);
        let dev = g.dev.clone();
        let cfg = config(&snapshot());
        let props = cfg.props();
        let (kh, vh) = (props.num_k_heads, props.num_v_heads);
        let (vb, names) = Recorder::over(&weights(&snapshot()), DType::BF16, &dev)?;
        let gdn = QGatedDeltaNet::new(&vb.pp("model.language_model.layers.0.linear_attn"), &props, &cfg.quant()?)?;
        let key_dim = kh * props.head_k_dim;
        let conv_dim = 2 * key_dim + vh * props.head_v_dim;
        let mut cache = GdnLayerCache {
            conv_state: Tensor::zeros((1, conv_dim, props.conv_kernel), DType::BF16, &dev)?,
            recurrent_state: Tensor::zeros((1, vh, props.head_k_dim, props.head_v_dim), DType::F32, &dev)?,
            seqlen_offset: 0,
            trail: None,
        };
        let u = fixture("l0.attn.u");
        let mut stages: HashMap<&str, Vec<Tensor>> = HashMap::new();
        let mut outs = vec![];
        let mut at = 0;
        for len in CHUNKS {
            let mut probe = vec![];
            outs.push(gdn.forward_probed(&chunk(&u, at, len, &dev), &mut cache, Some(&mut probe))?.squeeze(0)?);
            for (name, t) in probe {
                stages.entry(name).or_default().push(t.squeeze(0)?.reshape((len, ()))?);
            }
            at += len;
        }
        let cat = |name: &str| Tensor::cat(&stages[name], 0).expect("cat");
        let conv = grouped(&cat("conv"), 1, 2 * key_dim, vh, kh, props.head_v_dim);
        assert_close("gdn.conv vs served", &conv, &vectors("gdn.prefill.conv"), Tol::ulps(1.0));
        // the first K tiled heads are the key heads themselves
        let first = |t: Tensor| t.narrow(1, 0, key_dim).expect("key heads");
        assert_close("gdn.q vs served", &first(cat("q")), &vectors("gdn.prefill.q"), Tol::ulps(1.0));
        assert_close("gdn.k vs served", &first(cat("k")), &vectors("gdn.prefill.k"), Tol::ulps(1.0));
        let v = grouped(&cat("v"), 1, 0, vh, kh, props.head_v_dim);
        assert_close("gdn.v vs served", &v, &vectors("gdn.prefill.v"), Tol::ulps(1.0));
        let rel = |what: &str, got: Tensor, want: Tensor| {
            let err = (got - &want).and_then(|d| d.abs()).and_then(|d| d.max_all()).and_then(|d| d.to_scalar::<f32>()).unwrap();
            let scale = want.abs().and_then(|d| d.max_all()).and_then(|d| d.to_scalar::<f32>()).unwrap();
            println!("{what}: max relative {:e}", err / scale);
            assert!(err <= 1e-6 * scale.max(1.0), "{what}: {err:e} of {scale:e}");
        };
        rel("gdn.g", grouped(&cat("g"), 1, 0, vh, kh, 1).to_device(&hanzo_ml::Device::Cpu)?, vectors("gdn.prefill.g"));
        rel("gdn.beta", grouped(&cat("beta"), 1, 0, vh, kh, 1).to_device(&hanzo_ml::Device::Cpu)?, vectors("gdn.prefill.beta"));
        let core = grouped(&cat("core"), 1, 0, vh, kh, props.head_v_dim);
        assert_close("gdn.core vs served", &core, &vectors("gdn.prefill.core"), Tol::most(8.0, 0.999, 0.9999));
        assert_close("gdn.core vs golden", &core, &fixture("l0.gdn.core"), Tol::most(4.0, 0.999, 0.0));
        let normed = grouped(&cat("normed"), 1, 0, vh, kh, props.head_v_dim);
        assert_close("gdn.normed token 24 vs served decode", &normed.narrow(0, 23, 1)?, &vectors("gdn.decode1.normed"), Tol::most(8.0, 0.999, 0.0));
        let out = Tensor::cat(&outs, 0)?;
        assert_close("gdn block vs golden", &out, &fixture("l0.attn.y"), Tol::most(4.0, 0.999, 0.9999));

        // the final state, grouped, every fourth head as the golden keeps it
        let state = grouped(&cache.recurrent_state.squeeze(0)?, 0, 0, vh, kh, 1);
        let sub: Vec<Tensor> = (0..vh).step_by(4).map(|j| state.get(j).unwrap()).collect();
        let sub = Tensor::stack(&sub, 0)?.to_device(&hanzo_ml::Device::Cpu)?;
        let want = fixture("l0.gdn.state");
        let err = (sub - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
        println!("gdn state: max relative {:e}", err / scale);
        assert!(err <= 1e-4 * scale, "state {err:e} of {scale:e}");

        let want: std::collections::BTreeSet<String> = manifest(cfg.text())?
            .into_iter()
            .map(|e| e.name)
            .filter(|n| n.starts_with(&format!("{PREFIX}layers.0.linear_attn.")))
            .collect();
        assert_eq!(*names.lock().unwrap(), want, "tensors read");
        Ok(())
    }

    /// Layer 3's MoE in chunks [16, 7, 1]: the router, the routing, the NVFP4 experts against
    /// FlashInfer's CUTLASS MoE, the shared expert against cutlass_scaled_mm, and the block.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot; needs a CUDA device"]
    fn golden_moe() -> Result<()> {
        use crate::models::quantized_qwen3_5_moe::FusedMoe;
        use hanzo_ml::Tensor;
        use hanzo_quant::QuantMethod;
        let g = gpu(1.45);
        let dev = g.dev.clone();
        let cfg = config(&snapshot());
        let props = cfg.props();
        let (vb, names) = Recorder::over(&weights(&snapshot()), DType::BF16, &dev)?;
        let moe = FusedMoe::new(&vb.pp("model.language_model.layers.3.mlp"), &props, &cfg.quant()?, DType::BF16)?;
        let u = fixture("l3.mlp.u");
        let (ids_ref, w_ref) = (fixture("l3.ids"), fixture("l3.weights"));
        let (mut logits, mut routed, mut shared, mut outs, mut own_ids) = (vec![], vec![], vec![], vec![], vec![]);
        let mut at = 0;
        for len in CHUNKS {
            let x = chunk(&u, at, len, &dev).squeeze(0)?;
            logits.push(moe.logits(&x)?);
            let ids = ids_ref.narrow(0, at, len)?.to_dtype(DType::U32)?.to_device(&dev)?;
            let w = w_ref.narrow(0, at, len)?.to_device(&dev)?;
            routed.push(moe.experts(&x, &ids, &w)?);
            shared.push(moe.shared(&x)?);
            let (own, _) = moe.route(&moe.logits(&x)?)?;
            own_ids.push(own.to_dtype(DType::U32)?);
            outs.push(moe.forward(&x.unsqueeze(0)?)?.squeeze(0)?);
            at += len;
        }
        let cat = |v: &[Tensor]| Tensor::cat(v, 0).expect("cat");
        assert_close("router logits", &cat(&logits), &fixture("l3.router"), Tol::ulps(1.0));
        // routing from the reference logits
        let (ids, w) = moe.route(&fixture("l3.router").to_device(&dev)?)?;
        assert_eq!(
            ids.to_dtype(DType::U32)?.to_device(&hanzo_ml::Device::Cpu)?.to_vec2::<u32>()?,
            ids_ref.to_dtype(DType::U32)?.to_vec2::<u32>()?,
            "expert ids from the reference logits"
        );
        let werr = (w.to_device(&hanzo_ml::Device::Cpu)? - &w_ref)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(werr <= 1e-6, "routing weights {werr:e}");
        // fc1's activation quantizer against vLLM's scaled_fp4_quant
        let gs = fixture("l3.mlp.gs").to_vec1::<f32>()?;
        let (codes, scales) = hanzo_quant::quantize::nvfp4(&u.to_device(&dev)?, gs[0])?;
        let bytes = |t: &Tensor| -> Vec<u8> {
            let t = t.to_device(&hanzo_ml::Device::Cpu).unwrap();
            match t.dtype() {
                DType::U8 => t.flatten_all().unwrap().to_vec1::<u8>().unwrap(),
                _ => fp8_codes(&t),
            }
        };
        assert_eq!(bytes(&codes), bytes(&quant_fixture("fp4.input.l3mu.q")), "fc1 input codes");
        assert_eq!(bytes(&scales), bytes(&quant_fixture("fp4.input.l3mu.s")), "fc1 input scales");
        let routed = cat(&routed);
        assert_close("experts vs served", &routed, &vectors("moe.routed"), Tol::most(2.0, 0.999, 0.99999));
        assert_close("experts vs golden", &routed, &fixture("l3.routed"), Tol::most(1.0, 0.999, 0.0));
        assert_close("shared vs golden", &cat(&shared), &fixture("l3.shared"), Tol::ulps(1.0));
        let down = moe.shared_down_proj.forward(&fixture("l3.shared.act").to_device(&dev)?)?;
        assert_close("shared down vs served w8a8", &down, &vectors("w8a8.shared_down"), Tol::most(1.0, 0.999, 0.0));
        assert_eq!(
            cat(&own_ids).to_device(&hanzo_ml::Device::Cpu)?.to_vec2::<u32>()?,
            ids_ref.to_dtype(DType::U32)?.to_vec2::<u32>()?,
            "own routing"
        );
        assert_close("moe block vs golden", &cat(&outs), &fixture("l3.moe"), Tol::most(4.0, 1.0, 0.9999));
        let want: std::collections::BTreeSet<String> = manifest(cfg.text())?
            .into_iter()
            .map(|e| e.name)
            .filter(|n| n.starts_with(&format!("{PREFIX}layers.3.mlp.")))
            .collect();
        assert_eq!(*names.lock().unwrap(), want, "tensors read");
        Ok(())
    }

    /// The hash read from the checkpoint's I64 buffers gives the GGUF's constants.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot"]
    fn hash_new_matches_gguf_constants() -> Result<()> {
        let cfg = config(&snapshot());
        let vb = vb(&hanzo_ml::Device::Cpu)?;
        let t = cfg.text();
        let hash = crate::models::ngram::Hash::new(
            &vb.pp(format!("{PREFIX}layers.{}.ple.ple_embedding", t.ple_layer()?)),
            t.eos_token_id,
            t.heads_per_ngram,
        )?;
        assert_eq!(
            hash.rows(&[], &[9707, 11, 1879])[32..],
            [
                6380558, 26411572, 56460672, 78566983, 94693008, 106742196, 124822692, 148942556,
                164226950, 190352573, 210933682, 238908951, 242182004, 265475238, 299910982,
                312121804
            ]
        );
        Ok(())
    }

    /// Layer 1's n-gram block in chunks [16, 7, 1] with its conv history carried: ids, table
    /// bytes and embedding exact, every stored stage within 1 ulp, the streams after the add.
    #[test]
    #[ignore = "reads the Qwen3.8-Flash-Next snapshot; needs a CUDA device"]
    fn golden_ple() -> Result<()> {
        use crate::models::gdn::GdnLayerCache;
        use crate::models::ngram::{Fp8Table, Hash, Ngram};
        use hanzo_ml::Tensor;
        let g = gpu(0.2);
        let dev = g.dev.clone();
        let cfg = config(&snapshot());
        let t = cfg.text();
        let layer = t.ple_layer()?;
        let files = weights(&snapshot());
        let (vb, names) = Recorder::over(&files, DType::BF16, &dev)?;
        let pre = format!("{PREFIX}layers.{layer}.ple");
        let hash = Hash::new(&vb.pp(format!("{pre}.ple_embedding")), t.eos_token_id, t.heads_per_ngram)?;
        let table = Fp8Table::open(&files, &format!("{pre}.ple_embedding.ngram_embedding"))?;
        let ngram = Ngram::new(&vb.pp(&pre), hash, table, t.rms_norm_eps)?;
        let (c, rows) = ngram.history()?;
        let mut cache = GdnLayerCache {
            conv_state: Tensor::zeros((1, c, rows), DType::BF16, &dev)?,
            recurrent_state: Tensor::zeros(1, DType::F32, &dev)?,
            seqlen_offset: 0,
            trail: None,
        };
        let toks = tokens();
        let x = fixture("l1.x");
        let (mut ids, mut bytes, mut es, mut stages, mut xps) = (vec![], vec![], vec![], HashMap::<&str, Vec<Tensor>>::new(), vec![]);
        let mut at: usize = 0;
        for len in CHUNKS {
            let prior = toks[at.saturating_sub(2)..at].to_vec();
            let chunk_ids = toks[at..at + len].to_vec();
            ids.extend(ngram.rows(&prior, &chunk_ids));
            bytes.extend(ngram.row_bytes(&prior, &chunk_ids)?);
            let e = ngram.embed(&[prior], &[chunk_ids], &dev)?;
            es.push(e.squeeze(0)?);
            let xs = chunk(&x, at, len, &dev).reshape((1, len, 4, t.hidden_size))?;
            let mut probe = vec![];
            let delta = ngram.forward_probed(&xs, &e, &mut cache, Some(&mut probe))?;
            for (name, v) in probe {
                stages.entry(name).or_default().push(v.squeeze(0)?.reshape((len, ()))?);
            }
            let xp = (xs.to_dtype(DType::F32)? + delta)?.to_dtype(DType::BF16)?;
            xps.push(xp.reshape((len, ()))?);
            at += len;
        }
        let want_ids: Vec<u32> = fixture("l1.ple.ids").flatten_all()?.to_dtype(DType::U32)?.to_vec1()?;
        assert_eq!(ids, want_ids, "n-gram ids");
        let want_bytes: Vec<u8> = fixture("l1.ple.rows").flatten_all()?.to_vec1()?;
        assert_eq!(bytes, want_bytes, "table bytes");
        let e = Tensor::cat(&es, 0)?;
        assert_close("ple.e (bit-exact)", &e, &fixture("l1.ple.e"), Tol::ulps(0.0));
        let cat = |name: &str| Tensor::cat(&stages[name], 0).expect("cat");
        for (stage, want) in [("key", "l1.ple.key"), ("value", "l1.ple.value"), ("gv", "l1.ple.gv"), ("nrm", "l1.ple.nrm"), ("conv", "l1.ple.conv")] {
            assert_close(&format!("ple.{stage}"), &cat(stage), &fixture(want), Tol::ulps(1.0));
        }
        assert_close("streams after the n-gram add", &Tensor::cat(&xps, 0)?, &fixture("l1.xp"), Tol::most(4.0, 0.999, 0.9999));
        let want: std::collections::BTreeSet<String> = manifest(t)?
            .into_iter()
            .filter(|e| e.role == Role::Device || !e.name.contains(".shard_"))
            .map(|e| e.name)
            .filter(|n| n.starts_with(&format!("{pre}.")) && !n.ends_with("weight_scale"))
            .collect();
        assert_eq!(*names.lock().unwrap(), want, "tensors read through the varbuilder");
        Ok(())
    }

    #[test]
    fn deferrals_match_their_prefixes() {
        assert_eq!(
            deferred("model.language_model.layers.3.self_attn.indexer.k_layernorm.weight"),
            Some("self_attn.indexer.")
        );
        assert_eq!(deferred("mtp.layers.0.mlp.gate.weight"), Some("mtp."));
        assert_eq!(deferred("model.visual.blocks.0.norm1.weight"), Some("model.visual."));
        assert_eq!(deferred("model.language_model.layers.3.self_attn.q_proj.weight"), None);
    }
}
