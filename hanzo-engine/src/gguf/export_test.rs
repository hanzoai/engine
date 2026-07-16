//! Round-trip proof for GGUF export.
//!
//! Build a tiny llama-shaped model directory, export it via
//! `hanzo_quant::export_gguf`, then load the produced `.gguf` back through the
//! engine's own loader (`super::Content`) and assert (a) metadata keys parse,
//! (b) tensor shapes match, and (c) dequantized values match the source (F16
//! within tolerance; Q8_0 exactly against an independent quantize->dequantize).
//! Fully offline and deterministic — no network.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use hanzo_ml::quantized::gguf_file::Value;
use hanzo_ml::quantized::{GgmlDType, QTensor};
use hanzo_ml::{Device, Tensor};
use hanzo_quant::GgufOutputType;

use super::Content;

const HIDDEN: usize = 64;
const LAYERS: usize = 2;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const HEAD_DIM: usize = HIDDEN / HEADS; // 16
const KV_DIM: usize = KV_HEADS * HEAD_DIM; // 32
const FFN: usize = 128;
const VOCAB: usize = 32;

fn t2(r: usize, c: usize) -> Tensor {
    Tensor::rand(-0.1f32, 0.1f32, (r, c), &Device::Cpu).unwrap()
}

fn t1(n: usize) -> Tensor {
    Tensor::rand(-0.1f32, 0.1f32, (n,), &Device::Cpu).unwrap()
}

/// Write a tiny llama model (safetensors + config.json + tokenizer.json) into
/// `dir` and return the source tensors keyed by HF name for later comparison.
fn build_model(dir: &Path) -> HashMap<String, Tensor> {
    let mut st: HashMap<String, Tensor> = HashMap::new();
    st.insert("model.embed_tokens.weight".into(), t2(VOCAB, HIDDEN));
    st.insert("model.norm.weight".into(), t1(HIDDEN));
    st.insert("lm_head.weight".into(), t2(VOCAB, HIDDEN));
    for i in 0..LAYERS {
        let p = format!("model.layers.{i}");
        st.insert(
            format!("{p}.self_attn.q_proj.weight"),
            t2(HEADS * HEAD_DIM, HIDDEN),
        );
        st.insert(format!("{p}.self_attn.k_proj.weight"), t2(KV_DIM, HIDDEN));
        st.insert(format!("{p}.self_attn.v_proj.weight"), t2(KV_DIM, HIDDEN));
        st.insert(
            format!("{p}.self_attn.o_proj.weight"),
            t2(HIDDEN, HEADS * HEAD_DIM),
        );
        st.insert(format!("{p}.mlp.gate_proj.weight"), t2(FFN, HIDDEN));
        st.insert(format!("{p}.mlp.up_proj.weight"), t2(FFN, HIDDEN));
        st.insert(format!("{p}.mlp.down_proj.weight"), t2(HIDDEN, FFN));
        st.insert(format!("{p}.input_layernorm.weight"), t1(HIDDEN));
        st.insert(format!("{p}.post_attention_layernorm.weight"), t1(HIDDEN));
    }
    hanzo_ml::safetensors::save(&st, dir.join("model.safetensors")).unwrap();
    fs::write(dir.join("config.json"), config_json()).unwrap();
    fs::write(dir.join("tokenizer.json"), tokenizer_json()).unwrap();
    st
}

fn config_json() -> String {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "_name_or_path": "tiny-llama-test",
        "hidden_size": HIDDEN,
        "num_hidden_layers": LAYERS,
        "num_attention_heads": HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "intermediate_size": FFN,
        "vocab_size": VOCAB,
        "max_position_embeddings": 128,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": false
    })
    .to_string()
}

fn tokenizer_json() -> String {
    let mut vocab = serde_json::Map::new();
    for (tok, id) in [("<unk>", 0u64), ("<s>", 1), ("</s>", 2)] {
        vocab.insert(tok.into(), serde_json::Value::from(id));
    }
    for i in 3..VOCAB {
        vocab.insert(format!("t{i}"), serde_json::Value::from(i as u64));
    }
    serde_json::json!({
        "version": "1.0",
        "model": { "type": "BPE", "vocab": vocab, "merges": ["t3 t4"] },
        "added_tokens": [
            {"id": 0, "content": "<unk>", "special": true},
            {"id": 1, "content": "<s>", "special": true},
            {"id": 2, "content": "</s>", "special": true}
        ]
    })
    .to_string()
}

fn max_abs_diff(a: &Tensor, b: &Tensor) -> f32 {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    a.iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn assert_common_metadata(md: &HashMap<String, Value>) {
    // Compare in the widening direction (u32 -> usize) to avoid lossy casts.
    let u = |k: &str| md[k].to_u32().unwrap() as usize;
    assert_eq!(
        md["general.architecture"].to_string().unwrap().as_str(),
        "llama"
    );
    assert_eq!(u("llama.block_count"), LAYERS);
    assert_eq!(u("llama.embedding_length"), HIDDEN);
    assert_eq!(u("llama.feed_forward_length"), FFN);
    assert_eq!(u("llama.attention.head_count"), HEADS);
    assert_eq!(u("llama.attention.head_count_kv"), KV_HEADS);
    assert_eq!(u("llama.rope.dimension_count"), HEAD_DIM);
    assert!(
        (md["llama.attention.layer_norm_rms_epsilon"]
            .to_f32()
            .unwrap()
            - 1e-5)
            .abs()
            < 1e-9
    );
    assert_eq!(
        md["tokenizer.ggml.model"].to_string().unwrap().as_str(),
        "gpt2"
    );
    assert!(md.contains_key("tokenizer.ggml.eos_token_id"));
    match &md["tokenizer.ggml.tokens"] {
        Value::Array(toks) => assert_eq!(toks.len(), VOCAB),
        other => panic!("tokens is not an array: {other:?}"),
    }
}

#[test]
fn gguf_export_roundtrip_f16() {
    let tmp = tempfile::tempdir().unwrap();
    let src = build_model(tmp.path());
    let out = tmp.path().join("model-f16.gguf");
    hanzo_quant::export_gguf(tmp.path().to_str().unwrap(), &out, GgufOutputType::F16).unwrap();

    let mut f = fs::File::open(&out).unwrap();
    let mut readers = [&mut f];
    let mut ct = Content::from_readers(&mut readers).unwrap();

    // (a) metadata keys parse.
    assert_common_metadata(ct.get_metadata());

    let dev = Device::Cpu;
    // (b) shapes + (c) F16 values within tolerance.
    let q = ct.tensor("blk.0.attn_q.weight", &dev).unwrap();
    assert_eq!(q.dtype(), GgmlDType::F16);
    assert_eq!(q.shape().dims(), &[HIDDEN, HIDDEN]);
    let deq = q.dequantize(&dev).unwrap();
    let want = &src["model.layers.0.self_attn.q_proj.weight"];
    assert!(
        max_abs_diff(want, &deq) < 4e-3,
        "F16 attn_q outside tolerance"
    );

    let down = ct.tensor("blk.1.ffn_down.weight", &dev).unwrap();
    assert_eq!(down.shape().dims(), &[HIDDEN, FFN]);

    // Norms stay F32 and are bit-exact.
    let n = ct.tensor("output_norm.weight", &dev).unwrap();
    assert_eq!(n.dtype(), GgmlDType::F32);
    assert_eq!(n.shape().dims(), &[HIDDEN]);
    assert_eq!(
        max_abs_diff(&src["model.norm.weight"], &n.dequantize(&dev).unwrap()),
        0.0
    );
}

#[test]
fn gguf_export_roundtrip_q8_0() {
    let tmp = tempfile::tempdir().unwrap();
    let src = build_model(tmp.path());
    let out = tmp.path().join("model-q8_0.gguf");
    hanzo_quant::export_gguf(tmp.path().to_str().unwrap(), &out, GgufOutputType::Q8_0).unwrap();

    let mut f = fs::File::open(&out).unwrap();
    let mut readers = [&mut f];
    let mut ct = Content::from_readers(&mut readers).unwrap();

    assert_common_metadata(ct.get_metadata());

    // 2-D weights are Q8_0; token_embd too.
    assert_eq!(
        ct.tensor_info("token_embd.weight").unwrap().ggml_dtype,
        GgmlDType::Q8_0
    );
    assert_eq!(
        ct.tensor_info("blk.1.ffn_down.weight").unwrap().ggml_dtype,
        GgmlDType::Q8_0
    );
    // 1-D norms stay F32.
    assert_eq!(
        ct.tensor_info("blk.0.attn_norm.weight").unwrap().ggml_dtype,
        GgmlDType::F32
    );

    let dev = Device::Cpu;
    // (b) shape + (c) Q8_0 exact against an independent quantize->dequantize.
    let down = ct.tensor("blk.1.ffn_down.weight", &dev).unwrap();
    assert_eq!(down.shape().dims(), &[HIDDEN, FFN]);
    let deq = down.dequantize(&dev).unwrap();
    let want = &src["model.layers.1.mlp.down_proj.weight"];
    let reference = QTensor::quantize(want, GgmlDType::Q8_0)
        .unwrap()
        .dequantize(&dev)
        .unwrap();
    assert_eq!(
        max_abs_diff(&reference, &deq),
        0.0,
        "Q8_0 ffn_down mismatch"
    );
}
