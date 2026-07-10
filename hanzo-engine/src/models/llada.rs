//! LLaDA: a masked diffusion language model (GSAI-ML/LLaDA-8B).
//!
//! Architecture is a Llama-style pre-norm transformer (RMSNorm, RoPE, split
//! SwiGLU) with two differences from a causal decoder: attention is
//! BIDIRECTIONAL (no causal mask) and generation is by iterative masked
//! denoising rather than token-append. There is no KV cache: every denoise
//! step re-runs the full sequence. See `modeling_llada.py` (LLaDALlamaBlock).

use hanzo_ml::{DType, Device, IndexOp, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::{QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder};
use serde::Deserialize;
use std::sync::Arc;

use crate::attention::{AttentionMask, SdpaParams};
use crate::layers::{embedding, Activation, RmsNorm, RotaryEmbedding, Sdpa};
use crate::pipeline::text_models_inputs_processor::FlashParams;
use crate::utils::progress::NiceProgressBar;
use crate::utils::varbuilder_utils::{from_mmaped_safetensors, DeviceForLoadTensor};

// LLaDA uses the OLMo config schema (d_model/n_heads/mlp_hidden_size), not HF-llama names.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_kv_heads: Option<usize>,
    pub n_layers: usize,
    pub mlp_hidden_size: usize,
    pub vocab_size: usize,
    pub embedding_size: Option<usize>,
    pub rms_norm_eps: f64,
    pub rope_theta: f32,
    pub max_sequence_length: usize,
    pub mask_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub include_bias: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
    pub fn num_kv_heads(&self) -> usize {
        self.n_kv_heads.unwrap_or(self.n_heads)
    }
    pub fn embed_size(&self) -> usize {
        self.embedding_size.unwrap_or(self.vocab_size)
    }
}

struct Attention {
    q_proj: Arc<dyn QuantMethod>,
    k_proj: Arc<dyn QuantMethod>,
    v_proj: Arc<dyn QuantMethod>,
    attn_out: Arc<dyn QuantMethod>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    sdpa_params: SdpaParams,
}

impl Attention {
    fn load(vb: ShardedVarBuilder, cfg: &Config, rotary: Arc<RotaryEmbedding>) -> Result<Self> {
        let hidden = cfg.d_model;
        let head_dim = cfg.head_dim();
        let q_out = cfg.n_heads * head_dim;
        let kv_out = cfg.num_kv_heads() * head_dim;
        let quant = &cfg.quantization_config;
        let q_proj = ReplicatedLayer::new(hidden, q_out, quant, cfg.include_bias, vb.pp("q_proj"))?;
        let k_proj = ReplicatedLayer::new(hidden, kv_out, quant, cfg.include_bias, vb.pp("k_proj"))?;
        let v_proj = ReplicatedLayer::new(hidden, kv_out, quant, cfg.include_bias, vb.pp("v_proj"))?;
        let attn_out =
            ReplicatedLayer::new(q_out, hidden, quant, cfg.include_bias, vb.pp("attn_out"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            attn_out,
            num_heads: cfg.n_heads,
            num_kv_heads: cfg.num_kv_heads(),
            head_dim,
            rotary,
            sdpa_params: SdpaParams {
                n_kv_groups: cfg.n_heads / cfg.num_kv_heads(),
                softcap: None,
                softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                sliding_window: None,
                sinks: None,
            },
        })
    }

    // Bidirectional attention over the whole sequence, no KV cache.
    fn forward(&self, x: &Tensor, positions: &Tensor, bidir: &FlashParams) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let (q, k) = self.rotary.forward_positions(&q, &k, positions)?;

        // AttentionMask::None + FlashParams{causal:false} => full bidirectional (pitfall 6).
        let y = Sdpa.run_attention(
            &q,
            &k,
            &v,
            &AttentionMask::None,
            Some(bidir),
            &self.sdpa_params,
        )?;
        let y = y.transpose(1, 2)?.reshape((b, l, ()))?;
        self.attn_out.forward(&y)
    }
}

struct Mlp {
    ff_proj: Arc<dyn QuantMethod>,
    up_proj: Arc<dyn QuantMethod>,
    ff_out: Arc<dyn QuantMethod>,
    act: Activation,
}

impl Mlp {
    fn load(vb: ShardedVarBuilder, cfg: &Config) -> Result<Self> {
        let hidden = cfg.d_model;
        let inter = cfg.mlp_hidden_size;
        let quant = &cfg.quantization_config;
        Ok(Self {
            ff_proj: ReplicatedLayer::new(hidden, inter, quant, cfg.include_bias, vb.pp("ff_proj"))?,
            up_proj: ReplicatedLayer::new(hidden, inter, quant, cfg.include_bias, vb.pp("up_proj"))?,
            ff_out: ReplicatedLayer::new(inter, hidden, quant, cfg.include_bias, vb.pp("ff_out"))?,
            act: Activation::Silu,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.act.forward(&self.ff_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.ff_out.forward(&(gate * up)?)
    }
}

struct Block {
    attn_norm: RmsNorm,
    attn: Attention,
    ff_norm: RmsNorm,
    mlp: Mlp,
}

impl Block {
    fn load(vb: ShardedVarBuilder, cfg: &Config, rotary: Arc<RotaryEmbedding>) -> Result<Self> {
        Ok(Self {
            attn_norm: RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("attn_norm"))?,
            attn: Attention::load(vb.clone(), cfg, rotary)?,
            ff_norm: RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb.pp("ff_norm"))?,
            mlp: Mlp::load(vb.clone(), cfg)?,
        })
    }

    fn forward(&self, x: &Tensor, positions: &Tensor, bidir: &FlashParams) -> Result<Tensor> {
        let x = (x + self.attn.forward(&self.attn_norm.forward(x)?, positions, bidir)?)?;
        let out = (&x + self.mlp.forward(&self.ff_norm.forward(&x)?)?)?;
        Ok(out)
    }
}

pub struct Model {
    wte: Embedding,
    blocks: Vec<Block>,
    ln_f: RmsNorm,
    ff_out: Arc<dyn QuantMethod>,
    device: Device,
    dtype: DType,
    pub cfg: Config,
}

impl Model {
    pub fn new(
        cfg: &Config,
        vb: ShardedVarBuilder,
        device: &Device,
        multi_progress: &Arc<indicatif::MultiProgress>,
    ) -> Result<Self> {
        let vb_t = vb.pp("model").pp("transformer");
        let wte = embedding(
            cfg.embed_size(),
            cfg.d_model,
            vb_t.pp("wte"),
            &cfg.quantization_config,
        )?;
        let ln_f = RmsNorm::new(cfg.d_model, cfg.rms_norm_eps, vb_t.pp("ln_f"))?;
        let ff_out = ReplicatedLayer::new(
            cfg.d_model,
            cfg.embed_size(),
            &cfg.quantization_config,
            false,
            vb_t.pp("ff_out"),
        )?;
        let rotary = Arc::new(RotaryEmbedding::new(
            cfg.rope_theta,
            cfg.head_dim(),
            cfg.max_sequence_length,
            device,
            true,
            vb.dtype(),
        )?);
        let vb_blocks = vb_t.pp("blocks");
        let blocks = NiceProgressBar::<_, 'b'>(0..cfg.n_layers, "Loading repeating layers", multi_progress)
            .par_iter_if_isq(|i| Block::load(vb_blocks.pp(i), cfg, rotary.clone()))?;
        Ok(Self {
            wte,
            blocks,
            ln_f,
            ff_out,
            device: device.clone(),
            dtype: vb.dtype(),
            cfg: cfg.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    pub fn max_seq_len(&self) -> usize {
        self.cfg.max_sequence_length
    }

    // Full-sequence logits [B, L, vocab] at every position (diffusion needs all positions).
    // `seqlen_offsets` are per-sequence RoPE start offsets (all-zero for a fresh sequence).
    pub fn forward(&self, input_ids: &Tensor, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let bidir = FlashParams::empty(false);
        let positions = self.rope_positions(seqlen_offsets)?;
        let mut x = self.wte.forward(input_ids)?;
        for block in &self.blocks {
            x = block.forward(&x, &positions, &bidir)?;
        }
        let x = self.ln_f.forward(&x)?;
        self.ff_out.forward(&x)
    }

    fn rope_positions(&self, seqlen_offsets: &[usize]) -> Result<Tensor> {
        let positions: Vec<u32> = seqlen_offsets.iter().map(|&o| o as u32).collect();
        let n = positions.len();
        Tensor::from_vec(positions, n, &self.device)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GenParams {
    pub gen_length: usize,
    pub steps: usize,
    pub block_length: usize,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            gen_length: 128,
            steps: 128,
            block_length: 32,
        }
    }
}

// Linear noise schedule: how many masked tokens to unmask at each step (base + a
// leading remainder), summing to the block's mask count. See LLaDA generate.py.
fn num_transfer_tokens(mask_count: usize, steps: usize) -> Vec<usize> {
    let base = mask_count / steps;
    let rem = mask_count % steps;
    (0..steps).map(|i| base + usize::from(i < rem)).collect()
}

impl Model {
    // Masked-diffusion generation: semi-autoregressive blocks, greedy (temperature 0),
    // low-confidence remasking. Deterministic. Returns the gen_length response tokens.
    pub fn generate(&self, prompt: &[u32], params: &GenParams) -> Result<Vec<u32>> {
        let mask = self.cfg.mask_token_id;
        let prompt_len = prompt.len();
        let total = prompt_len + params.gen_length;
        let mut x = Vec::with_capacity(total);
        x.extend_from_slice(prompt);
        x.extend(std::iter::repeat(mask).take(params.gen_length));

        let num_blocks = params.gen_length / params.block_length;
        let steps_per_block = params.steps / num_blocks;
        let offsets = [0usize];

        for nb in 0..num_blocks {
            let block_end = prompt_len + (nb + 1) * params.block_length;
            let schedule = num_transfer_tokens(params.block_length, steps_per_block);
            for &k in &schedule {
                let input = Tensor::from_vec(x.clone(), (1, total), &self.device)?;
                let logits = self.forward(&input, &offsets)?.i(0)?.to_dtype(DType::F32)?;
                let x0 = logits.argmax(D::Minus1)?.to_vec1::<u32>()?;
                let conf = hanzo_nn::ops::softmax_last_dim(&logits)?
                    .max(D::Minus1)?
                    .to_vec1::<f32>()?;

                let mut cands: Vec<(usize, f32)> = (0..block_end)
                    .filter(|&p| x[p] == mask)
                    .map(|p| (p, conf[p]))
                    .collect();
                cands.sort_by(|a, b| b.1.total_cmp(&a.1));
                for &(p, _) in cands.iter().take(k.min(cands.len())) {
                    x[p] = x0[p];
                }
            }
        }
        Ok(x[prompt_len..].to_vec())
    }
}

pub(crate) fn load_from_dir(
    dir: &std::path::Path,
    device: &Device,
    dtype: DType,
) -> Result<Model> {
    let cfg_str = std::fs::read_to_string(dir.join("config.json")).map_err(hanzo_ml::Error::msg)?;
    let cfg: Config = serde_json::from_str(&cfg_str).map_err(hanzo_ml::Error::msg)?;
    let mut st: Vec<std::path::PathBuf> = std::fs::read_dir(dir)
        .map_err(hanzo_ml::Error::msg)?
        .filter_map(|e| {
            let p = e.ok()?.path();
            (p.extension().is_some_and(|x| x == "safetensors")).then_some(p)
        })
        .collect();
    st.sort();
    let vb = from_mmaped_safetensors(
        st,
        vec![],
        Some(dtype),
        device,
        vec![None],
        true,
        None,
        |_| true,
        Arc::new(|_| DeviceForLoadTensor::Base),
    )?;
    Model::new(&cfg, vb, device, &Arc::new(indicatif::MultiProgress::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Gated: set LLADA_WEIGHTS to the local model dir. Compares against oracle tensors
    // produced by scratchpad/oracle.py. Run: `cargo test -p hanzo-engine llada_parity -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn llada_parity() {
        let dir = std::env::var("LLADA_WEIGHTS").expect("set LLADA_WEIGHTS");
        let oracle = std::env::var("LLADA_ORACLE").expect("set LLADA_ORACLE (scratchpad prefix)");
        let device = Device::Cpu;
        let model = load_from_dir(std::path::Path::new(&dir), &device, DType::F32).unwrap();

        // Gate 1: single-forward logit parity.
        let ids = Tensor::read_npy(format!("{oracle}_ids.npy")).unwrap().to_dtype(DType::U32).unwrap();
        let ref_logits = Tensor::read_npy(format!("{oracle}_logits.npy")).unwrap();
        let logits = model.forward(&ids, &[0]).unwrap().i(0).unwrap().to_dtype(DType::F32).unwrap();
        let l = logits.dim(0).unwrap();
        let mut worst = 1.0f32;
        for pos in (l.saturating_sub(32))..l {
            let a = logits.i(pos).unwrap();
            let b = ref_logits.i(pos).unwrap();
            let cos = cosine(&a, &b);
            worst = worst.min(cos);
        }
        println!("gate1 worst masked-position cosine = {worst:.6}");
        assert!(worst > 0.999, "logit cosine {worst} below 0.999");

        // Gate 2: full greedy generation token match.
        let gen: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(format!("{oracle}_gen.json")).unwrap()).unwrap();
        let params = GenParams {
            gen_length: gen["gen"].as_u64().unwrap() as usize,
            steps: gen["steps"].as_u64().unwrap() as usize,
            block_length: gen["block"].as_u64().unwrap() as usize,
        };
        for (k, o) in gen["outs"].as_object().unwrap() {
            let prompt_ids: Vec<u32> = o["input_ids"].as_array().unwrap().iter()
                .map(|v| v.as_u64().unwrap() as u32).collect();
            let ref_toks: Vec<u32> = o["gen_tokens"].as_array().unwrap().iter()
                .map(|v| v.as_u64().unwrap() as u32).collect();
            let toks = model.generate(&prompt_ids, &params).unwrap();
            let matches = toks.iter().zip(&ref_toks).filter(|(a, b)| a == b).count();
            println!("gate2 prompt{k}: {}/{} tokens match", matches, ref_toks.len());
            assert_eq!(toks, ref_toks, "prompt{k} token mismatch");
        }
    }

    fn envu(k: &str, default: usize) -> usize {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
    }

    // Coherence + throughput smoke on real weights (no oracle). Env: LLADA_WEIGHTS,
    // LLADA_GEN/LLADA_STEPS/LLADA_BLOCK, LLADA_F32. Run on GPU:
    // `cargo test -p hanzo-engine --features cuda llada_smoke -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn llada_smoke() {
        let dir = std::env::var("LLADA_WEIGHTS").expect("set LLADA_WEIGHTS");
        let dtype = if std::env::var("LLADA_F32").is_ok() { DType::F32 } else { DType::BF16 };
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("device={device:?} dtype={dtype:?}");
        let model = load_from_dir(std::path::Path::new(&dir), &device, dtype).unwrap();
        let tok = tokenizers::Tokenizer::from_file(std::path::Path::new(&dir).join("tokenizer.json")).unwrap();
        let gen = envu("LLADA_GEN", 64);
        let params = GenParams { gen_length: gen, steps: envu("LLADA_STEPS", gen), block_length: envu("LLADA_BLOCK", 32) };
        println!("params: gen={} steps={} block={}", params.gen_length, params.steps, params.block_length);
        for q in [
            "What is the capital of France? Answer in one word.",
            "Write a haiku about the ocean.",
        ] {
            let s = format!("<|startoftext|><|start_header_id|>user<|end_header_id|>\n\n{q}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n");
            let ids: Vec<u32> = tok.encode(s, false).unwrap().get_ids().to_vec();
            let t0 = std::time::Instant::now();
            let out = model.generate(&ids, &params).unwrap();
            let dt = t0.elapsed().as_secs_f32();
            let text = tok.decode(&out, true).unwrap();
            let tps = params.gen_length as f32 / dt;
            let tpt = params.gen_length as f32 / params.steps as f32;
            println!("\nPROMPT: {q}\nGEN [{dt:.2}s | {tps:.1} tok/s | {:.2} tok/traversal]:\n{text}\n---", tpt);
        }
    }

    fn cosine(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_vec1::<f32>().unwrap();
        let b = b.to_vec1::<f32>().unwrap();
        let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }
}
