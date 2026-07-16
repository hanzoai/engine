//! GGUF export: write a loaded llama-family model to a llama.cpp-loadable `.gguf`.
//!
//! Companion to the engine's GGUF *reader* (`hanzo-engine/src/gguf`). The on-disk
//! format primitives (`gguf_file::write`, `QTensor::quantize`) live in `hanzo-ml`;
//! this module is the model -> GGUF mapping and is the one home for GGUF *creation*:
//! llama metadata keys, llama.cpp tensor names, and per-tensor output typing (norms
//! and other 1-D tensors stay F32; 2-D weights become F16 or Q8_0).

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    str::FromStr,
};

use hf_hub::api::sync::ApiBuilder;

use hanzo_ml::{
    quantized::{
        gguf_file::{self, Value},
        GgmlDType, QTensor,
    },
    DType, Device, Error, Result, Tensor,
};

use crate::safetensors::MmapedSafetensors;

/// Output tensor type for a GGUF export. 2-D weights are written in this type; 1-D
/// tensors (norms, biases) always stay F32, matching llama.cpp conventions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufOutputType {
    F16,
    Q8_0,
}

impl GgufOutputType {
    fn ggml(self) -> GgmlDType {
        match self {
            Self::F16 => GgmlDType::F16,
            Self::Q8_0 => GgmlDType::Q8_0,
        }
    }

    /// llama.cpp `general.file_type` (ftype) code.
    fn file_type(self) -> u32 {
        match self {
            Self::F16 => 1,
            Self::Q8_0 => 7,
        }
    }
}

impl FromStr for GgufOutputType {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" | "float16" => Ok(Self::F16),
            "q8_0" | "q8" => Ok(Self::Q8_0),
            other => Err(format!(
                "unsupported GGUF output type `{other}` (expected `f16` or `q8_0`)"
            )),
        }
    }
}

/// Export a llama-family model to `output` as a single `.gguf` file.
///
/// `model` is a local directory (containing `config.json`, `tokenizer.json`, and
/// `*.safetensors`) or a Hugging Face repo id, which is fetched into the HF cache.
pub fn export_gguf(model: &str, output: &Path, ty: GgufOutputType) -> Result<()> {
    let dir = resolve_model_dir(model)?;
    let cfg = read_config(&dir)?;
    let vocab = read_tokenizer(&dir)?;

    let st_paths = safetensor_paths(&dir)?;
    if st_paths.is_empty() {
        return Err(Error::msg(format!(
            "no `.safetensors` files in `{}`",
            dir.display()
        )));
    }
    // SAFETY: inherited from memmap; the files outlive the mapping within this call.
    let st = unsafe { MmapedSafetensors::multi(&st_paths)? };

    let dev = Device::Cpu;
    let mut tensors: Vec<(String, Tensor)> = Vec::new();
    for (hf_name, _) in st.tensors() {
        if let Some(gguf_name) = map_tensor_name(&hf_name) {
            let t = st.load(&hf_name, &dev, Some(DType::F32))?;
            tensors.push((gguf_name, t));
        }
    }
    if tensors.is_empty() {
        return Err(Error::msg(
            "no llama-family tensors found; expected a llama/mistral-architecture model",
        ));
    }
    tensors.sort_by(|a, b| a.0.cmp(&b.0));

    let metadata = build_metadata(&cfg, &vocab, ty)?;
    write_gguf(output, &metadata, &tensors, ty)
}

/// Quantize `tensors` to the export type and write the full GGUF file. 1-D tensors
/// stay F32; 2-D tensors become `ty` (Q8_0 falls back to F16 when a row is not a
/// multiple of the Q8_0 block size).
fn write_gguf(
    output: &Path,
    metadata: &[(&str, Value)],
    tensors: &[(String, Tensor)],
    ty: GgufOutputType,
) -> Result<()> {
    let mut qtensors: Vec<(String, QTensor)> = Vec::with_capacity(tensors.len());
    for (name, t) in tensors {
        let t = t.to_dtype(DType::F32)?;
        let dt = output_dtype(&t, ty);
        qtensors.push((name.clone(), QTensor::quantize(&t, dt)?));
    }

    let meta_ref: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
    let tensor_ref: Vec<(&str, &QTensor)> = qtensors.iter().map(|(k, t)| (k.as_str(), t)).collect();

    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut f = fs::File::create(output)?;
    gguf_file::write(&mut f, &meta_ref, &tensor_ref)
}

/// Per-tensor output dtype: 1-D tensors stay F32; 2-D tensors take the export type,
/// with Q8_0 falling back to F16 when the last dim is not block-aligned.
fn output_dtype(t: &Tensor, ty: GgufOutputType) -> GgmlDType {
    if t.rank() < 2 {
        return GgmlDType::F32;
    }
    if ty == GgufOutputType::Q8_0 {
        let last = t.dims()[t.rank() - 1];
        if !last.is_multiple_of(GgmlDType::Q8_0.block_size()) {
            return GgmlDType::F16;
        }
    }
    ty.ggml()
}

// ---------------------------------------------------------------------------
// Tensor name mapping: HF (safetensors) -> llama.cpp (GGUF).
// ---------------------------------------------------------------------------

fn map_tensor_name(hf: &str) -> Option<String> {
    match hf {
        "model.embed_tokens.weight" => return Some("token_embd.weight".to_string()),
        "model.norm.weight" => return Some("output_norm.weight".to_string()),
        "lm_head.weight" => return Some("output.weight".to_string()),
        _ => {}
    }

    let rest = hf.strip_prefix("model.layers.")?;
    let (idx, tail) = rest.split_once('.')?;
    if idx.is_empty() || !idx.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mapped = match tail {
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.k_proj.weight" => "attn_k.weight",
        "self_attn.v_proj.weight" => "attn_v.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "self_attn.q_proj.bias" => "attn_q.bias",
        "self_attn.k_proj.bias" => "attn_k.bias",
        "self_attn.v_proj.bias" => "attn_v.bias",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        _ => return None,
    };
    Some(format!("blk.{idx}.{mapped}"))
}

// ---------------------------------------------------------------------------
// config.json -> llama metadata.
// ---------------------------------------------------------------------------

struct LlamaConfig {
    name: String,
    context_length: u32,
    embedding_length: u32,
    block_count: u32,
    feed_forward_length: u32,
    head_count: u32,
    head_count_kv: u32,
    head_dim: u32,
    rms_norm_eps: f32,
    rope_freq_base: f32,
    vocab_size: u32,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
}

fn read_config(dir: &Path) -> Result<LlamaConfig> {
    let path = dir.join("config.json");
    let text = fs::read_to_string(&path)
        .map_err(|e| Error::msg(format!("reading {}: {e}", path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(Error::msg)?;

    let req_u32 = |key: &str| -> Result<u32> {
        v.get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as u32)
            .ok_or_else(|| Error::msg(format!("config.json missing required `{key}`")))
    };
    let opt_u32 = |key: &str| {
        v.get(key)
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as u32)
    };
    let opt_f32 = |key: &str| {
        v.get(key)
            .and_then(serde_json::Value::as_f64)
            .map(|n| n as f32)
    };

    let embedding_length = req_u32("hidden_size")?;
    let head_count = req_u32("num_attention_heads")?;
    let head_count_kv = opt_u32("num_key_value_heads").unwrap_or(head_count);
    let head_dim = opt_u32("head_dim").unwrap_or(embedding_length / head_count);

    // `bos_token_id` / `eos_token_id` may be a scalar or (for eos) a list; take the first.
    let first_id = |key: &str| -> Option<u32> {
        match v.get(key) {
            Some(serde_json::Value::Number(n)) => n.as_u64().map(|x| x as u32),
            Some(serde_json::Value::Array(a)) => a
                .first()
                .and_then(serde_json::Value::as_u64)
                .map(|x| x as u32),
            _ => None,
        }
    };

    let name = v
        .get("_name_or_path")
        .and_then(serde_json::Value::as_str)
        .or_else(|| v.get("model_type").and_then(serde_json::Value::as_str))
        .map(str::to_string)
        .unwrap_or_else(|| {
            dir.file_name()
                .and_then(|s| s.to_str())
                .unwrap_or("model")
                .to_string()
        });

    Ok(LlamaConfig {
        name,
        context_length: opt_u32("max_position_embeddings").unwrap_or(4096),
        embedding_length,
        block_count: req_u32("num_hidden_layers")?,
        feed_forward_length: req_u32("intermediate_size")?,
        head_count,
        head_count_kv,
        head_dim,
        rms_norm_eps: opt_f32("rms_norm_eps").unwrap_or(1e-5),
        rope_freq_base: opt_f32("rope_theta").unwrap_or(10_000.0),
        vocab_size: req_u32("vocab_size")?,
        bos_token_id: first_id("bos_token_id"),
        eos_token_id: first_id("eos_token_id"),
    })
}

// ---------------------------------------------------------------------------
// tokenizer.json -> tokenizer.ggml.* metadata.
// ---------------------------------------------------------------------------

/// llama.cpp token type codes.
const TT_NORMAL: i32 = 1;
const TT_UNKNOWN: i32 = 2;
const TT_CONTROL: i32 = 3;
const TT_USER_DEFINED: i32 = 4;
const TT_UNUSED: i32 = 5;

struct Vocab {
    model: String,
    tokens: Vec<String>,
    scores: Vec<f32>,
    token_type: Vec<i32>,
    merges: Vec<String>,
    unk_id: Option<u32>,
    bos_fallback: Option<u32>,
    eos_fallback: Option<u32>,
}

fn read_tokenizer(dir: &Path) -> Result<Vocab> {
    let path = dir.join("tokenizer.json");
    let text = fs::read_to_string(&path)
        .map_err(|e| Error::msg(format!("reading {}: {e}", path.display())))?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(Error::msg)?;
    let model = v
        .get("model")
        .ok_or_else(|| Error::msg("tokenizer.json missing `model`"))?;
    let kind = model
        .get("type")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("BPE");

    let mut vocab = if kind == "Unigram" {
        read_unigram(model)?
    } else {
        read_bpe(model)?
    };
    apply_added_tokens(&v, &mut vocab);

    // Special-token id fallbacks by content, used when config.json omits them.
    let find = |names: &[&str]| -> Option<u32> {
        vocab
            .tokens
            .iter()
            .position(|t| names.contains(&t.as_str()))
            .map(|i| i as u32)
    };
    vocab.bos_fallback = find(&["<s>", "<|startoftext|>", "<|begin_of_text|>"]);
    vocab.eos_fallback = find(&["</s>", "<|endoftext|>", "<|end_of_text|>", "<|im_end|>"]);
    Ok(vocab)
}

fn read_unigram(model: &serde_json::Value) -> Result<Vocab> {
    let entries = model
        .get("vocab")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| Error::msg("Unigram tokenizer missing `vocab` array"))?;
    let mut tokens = Vec::with_capacity(entries.len());
    let mut scores = Vec::with_capacity(entries.len());
    for e in entries {
        let pair = e
            .as_array()
            .ok_or_else(|| Error::msg("Unigram vocab entry is not a [token, score] pair"))?;
        tokens.push(
            pair.first()
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_string(),
        );
        scores.push(
            pair.get(1)
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0) as f32,
        );
    }
    let unk_id = model
        .get("unk_id")
        .and_then(serde_json::Value::as_u64)
        .map(|n| n as u32);
    let mut token_type = vec![TT_NORMAL; tokens.len()];
    if let Some(u) = unk_id {
        if let Some(t) = token_type.get_mut(u as usize) {
            *t = TT_UNKNOWN;
        }
    }
    Ok(Vocab {
        model: "llama".to_string(),
        tokens,
        scores,
        token_type,
        merges: Vec::new(),
        unk_id,
        bos_fallback: None,
        eos_fallback: None,
    })
}

fn read_bpe(model: &serde_json::Value) -> Result<Vocab> {
    let vocab_obj = model
        .get("vocab")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Error::msg("BPE tokenizer missing `vocab` object"))?;
    let max_id = vocab_obj
        .values()
        .filter_map(serde_json::Value::as_u64)
        .max()
        .map(|n| n as usize)
        .ok_or_else(|| Error::msg("BPE tokenizer has an empty `vocab`"))?;

    let mut tokens = vec![String::new(); max_id + 1];
    let mut present = vec![false; max_id + 1];
    for (tok, id) in vocab_obj {
        if let Some(id) = id.as_u64() {
            tokens[id as usize] = tok.clone();
            present[id as usize] = true;
        }
    }
    let mut token_type = vec![TT_NORMAL; tokens.len()];
    for (i, seen) in present.iter().enumerate() {
        if !*seen {
            tokens[i] = format!("[UNUSED{i}]");
            token_type[i] = TT_UNUSED;
        }
    }

    let merges = model
        .get("merges")
        .and_then(serde_json::Value::as_array)
        .map(|arr| arr.iter().filter_map(merge_to_string).collect())
        .unwrap_or_default();

    Ok(Vocab {
        model: "gpt2".to_string(),
        tokens,
        scores: Vec::new(),
        token_type,
        merges,
        unk_id: None,
        bos_fallback: None,
        eos_fallback: None,
    })
}

/// A merge is either `"a b"` (legacy) or `["a", "b"]` (current tokenizers).
fn merge_to_string(m: &serde_json::Value) -> Option<String> {
    match m {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Array(pair) if pair.len() == 2 => {
            let a = pair[0].as_str()?;
            let b = pair[1].as_str()?;
            Some(format!("{a} {b}"))
        }
        _ => None,
    }
}

fn apply_added_tokens(root: &serde_json::Value, vocab: &mut Vocab) {
    let Some(added) = root
        .get("added_tokens")
        .and_then(serde_json::Value::as_array)
    else {
        return;
    };
    for t in added {
        let Some(id) = t.get("id").and_then(serde_json::Value::as_u64) else {
            continue;
        };
        let id = id as usize;
        let content = t
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string();
        let special = t
            .get("special")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        if id >= vocab.tokens.len() {
            vocab.tokens.resize(id + 1, String::new());
            vocab.token_type.resize(id + 1, TT_UNUSED);
            if !vocab.scores.is_empty() {
                vocab.scores.resize(id + 1, 0.0);
            }
        }
        vocab.tokens[id] = content;
        vocab.token_type[id] = if special { TT_CONTROL } else { TT_USER_DEFINED };
    }
}

// ---------------------------------------------------------------------------
// Metadata assembly.
// ---------------------------------------------------------------------------

fn build_metadata(
    cfg: &LlamaConfig,
    vocab: &Vocab,
    ty: GgufOutputType,
) -> Result<Vec<(&'static str, Value)>> {
    let mut m: Vec<(&'static str, Value)> = vec![
        ("general.architecture", Value::String("llama".to_string())),
        ("general.name", Value::String(cfg.name.clone())),
        ("general.file_type", Value::U32(ty.file_type())),
        ("llama.context_length", Value::U32(cfg.context_length)),
        ("llama.embedding_length", Value::U32(cfg.embedding_length)),
        ("llama.block_count", Value::U32(cfg.block_count)),
        (
            "llama.feed_forward_length",
            Value::U32(cfg.feed_forward_length),
        ),
        ("llama.attention.head_count", Value::U32(cfg.head_count)),
        (
            "llama.attention.head_count_kv",
            Value::U32(cfg.head_count_kv),
        ),
        (
            "llama.attention.layer_norm_rms_epsilon",
            Value::F32(cfg.rms_norm_eps),
        ),
        ("llama.rope.freq_base", Value::F32(cfg.rope_freq_base)),
        ("llama.rope.dimension_count", Value::U32(cfg.head_dim)),
        ("llama.attention.key_length", Value::U32(cfg.head_dim)),
        ("llama.attention.value_length", Value::U32(cfg.head_dim)),
        ("llama.vocab_size", Value::U32(cfg.vocab_size)),
        ("tokenizer.ggml.model", Value::String(vocab.model.clone())),
        (
            "tokenizer.ggml.tokens",
            Value::Array(vocab.tokens.iter().cloned().map(Value::String).collect()),
        ),
        (
            "tokenizer.ggml.token_type",
            Value::Array(vocab.token_type.iter().map(|&t| Value::I32(t)).collect()),
        ),
    ];
    if !vocab.scores.is_empty() {
        m.push((
            "tokenizer.ggml.scores",
            Value::Array(vocab.scores.iter().map(|&s| Value::F32(s)).collect()),
        ));
    }
    if !vocab.merges.is_empty() {
        m.push((
            "tokenizer.ggml.merges",
            Value::Array(vocab.merges.iter().cloned().map(Value::String).collect()),
        ));
    }

    let eos = cfg
        .eos_token_id
        .or(vocab.eos_fallback)
        .ok_or_else(|| Error::msg("could not determine `eos_token_id` from config or tokenizer"))?;
    m.push(("tokenizer.ggml.eos_token_id", Value::U32(eos)));
    if let Some(bos) = cfg.bos_token_id.or(vocab.bos_fallback) {
        m.push(("tokenizer.ggml.bos_token_id", Value::U32(bos)));
    }
    if let Some(unk) = vocab.unk_id {
        m.push(("tokenizer.ggml.unknown_token_id", Value::U32(unk)));
    }
    Ok(m)
}

// ---------------------------------------------------------------------------
// Model source resolution.
// ---------------------------------------------------------------------------

fn safetensor_paths(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in
        fs::read_dir(dir).map_err(|e| Error::msg(format!("reading {}: {e}", dir.display())))?
    {
        let path = entry.map_err(Error::msg)?.path();
        if path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            paths.push(path);
        }
    }
    paths.sort();
    Ok(paths)
}

/// Return a local model directory: `model` itself when it is a directory, otherwise
/// the HF cache snapshot after fetching `config.json`, `tokenizer.json`, and the
/// model's safetensors shard(s).
fn resolve_model_dir(model: &str) -> Result<PathBuf> {
    let local = Path::new(model);
    if local.is_dir() {
        return Ok(local.to_path_buf());
    }

    let api = ApiBuilder::from_env()
        .with_progress(true)
        .build()
        .map_err(Error::msg)?;
    let repo = api.model(model.to_string());

    let cfg = repo.get("config.json").map_err(Error::msg)?;
    let dir = cfg
        .parent()
        .ok_or_else(|| Error::msg("HF cache path has no parent directory"))?
        .to_path_buf();
    repo.get("tokenizer.json").map_err(Error::msg)?;

    match repo.get("model.safetensors.index.json") {
        Ok(index) => {
            let text = fs::read_to_string(&index).map_err(Error::msg)?;
            let v: serde_json::Value = serde_json::from_str(&text).map_err(Error::msg)?;
            let mut shards = BTreeSet::new();
            if let Some(map) = v.get("weight_map").and_then(serde_json::Value::as_object) {
                for f in map.values() {
                    if let Some(s) = f.as_str() {
                        shards.insert(s.to_string());
                    }
                }
            }
            for shard in shards {
                repo.get(&shard).map_err(Error::msg)?;
            }
        }
        Err(_) => {
            repo.get("model.safetensors").map_err(Error::msg)?;
        }
    }
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::quantized::gguf_file::{Content, TensorInfo};
    use std::io::Cursor;

    fn roundtrip(
        metadata: &[(&str, Value)],
        tensors: &[(String, Tensor)],
        ty: GgufOutputType,
    ) -> (Content, Vec<u8>) {
        // Quantize + write through the same path `write_gguf` uses, but to memory.
        let mut qtensors: Vec<(String, QTensor)> = Vec::new();
        for (name, t) in tensors {
            let dt = output_dtype(t, ty);
            qtensors.push((name.clone(), QTensor::quantize(t, dt).unwrap()));
        }
        let meta_ref: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (*k, v)).collect();
        let tensor_ref: Vec<(&str, &QTensor)> =
            qtensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
        let mut buf = Cursor::new(Vec::new());
        gguf_file::write(&mut buf, &meta_ref, &tensor_ref).unwrap();
        let bytes = buf.into_inner();
        let mut cur = Cursor::new(bytes.clone());
        let content = Content::read(&mut cur).unwrap();
        (content, bytes)
    }

    fn read_back(content: &Content, bytes: &[u8], name: &str) -> Tensor {
        let info: &TensorInfo = content.tensor_infos.get(name).unwrap();
        let mut cur = Cursor::new(bytes.to_vec());
        info.read(&mut cur, content.tensor_data_offset, &Device::Cpu)
            .unwrap()
            .dequantize(&Device::Cpu)
            .unwrap()
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        a.iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn norm_stays_f32_and_2d_takes_export_type() {
        let dev = Device::Cpu;
        let norm = Tensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0], (4,), &dev).unwrap();
        let weight = Tensor::rand(-1.0f32, 1.0f32, (8, 64), &dev).unwrap();
        let tensors = vec![
            ("output_norm.weight".to_string(), norm.clone()),
            ("token_embd.weight".to_string(), weight.clone()),
        ];
        let meta = vec![("general.architecture", Value::String("llama".to_string()))];

        let (content, _) = roundtrip(&meta, &tensors, GgufOutputType::Q8_0);
        assert_eq!(
            content.tensor_infos["output_norm.weight"].ggml_dtype,
            GgmlDType::F32
        );
        assert_eq!(
            content.tensor_infos["token_embd.weight"].ggml_dtype,
            GgmlDType::Q8_0
        );
    }

    #[test]
    fn f16_roundtrip_matches_source_within_tolerance() {
        let dev = Device::Cpu;
        let weight = Tensor::rand(-1.0f32, 1.0f32, (8, 64), &dev).unwrap();
        let tensors = vec![("token_embd.weight".to_string(), weight.clone())];
        let meta = vec![("general.architecture", Value::String("llama".to_string()))];

        let (content, bytes) = roundtrip(&meta, &tensors, GgufOutputType::F16);
        assert_eq!(
            content.tensor_infos["token_embd.weight"].ggml_dtype,
            GgmlDType::F16
        );
        let back = read_back(&content, &bytes, "token_embd.weight");
        assert_eq!(back.dims(), &[8, 64]);
        // F16 dequant == source rounded to F16, i.e. within F16 precision.
        assert!(max_abs_diff(&weight, &back) < 4e-3);
    }

    #[test]
    fn q8_0_roundtrip_matches_reference_quantization() {
        let dev = Device::Cpu;
        let weight = Tensor::rand(-1.0f32, 1.0f32, (8, 64), &dev).unwrap();
        let tensors = vec![("token_embd.weight".to_string(), weight.clone())];
        let meta = vec![("general.architecture", Value::String("llama".to_string()))];

        let (content, bytes) = roundtrip(&meta, &tensors, GgufOutputType::Q8_0);
        let back = read_back(&content, &bytes, "token_embd.weight");
        assert_eq!(back.dims(), &[8, 64]);
        // Q8_0 dequant of the written file must equal an independent Q8_0
        // quantize -> dequantize of the same source, bit for bit.
        let reference = QTensor::quantize(&weight, GgmlDType::Q8_0)
            .unwrap()
            .dequantize(&dev)
            .unwrap();
        assert_eq!(max_abs_diff(&reference, &back), 0.0);
    }
}
