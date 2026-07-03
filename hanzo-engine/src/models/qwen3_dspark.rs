#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! DeepSpec **DSpark** parallel-block speculative draft model for Qwen3.
//!
//! A DSpark draft is NOT a normal decoder. It consumes several *target*
//! decoder-layer hidden states, fuses them into a fixed "context memory", seeds a
//! block of `block_size` slots (anchor token at slot 0, `mask_token_id` at the
//! rest), and — in a single forward through `num_hidden_layers` Qwen3 layers —
//! drafts the whole block at once. Each layer does cross-attention to the fixed
//! fused context (K/V projected per-layer from `target_hidden_states`) PLUS
//! bidirectional self-attention over the block. A vanilla Markov head then
//! autoregressively corrects the per-slot logits into a mutually-consistent
//! block of draft tokens, and a confidence head predicts a per-slot accept rate.
//!
//! Reference: `deepspec/modeling/dspark/{common.py,markov_head.py,qwen3/modeling.py}`.
//!
//! Notes on reuse:
//! * The Qwen3 decoder math (per-head q/k RMSNorm, GQA 32/8, head_dim 128, neox
//!   RoPE θ=1e6) is replicated here from first principles rather than through
//!   `models::qwen3::Attention` because that struct does kv-cache *self*-attention
//!   and cannot express DSpark's cross-attention-to-fixed-memory + block self-attn.
//! * The Markov step mirrors `hanzo_ml::markov_head::vanilla_sample_block`
//!   exactly. That module is the canonical home, but the engine's pinned
//!   `hanzo-ml 0.11.36` (crates.io) predates it, so the vanilla step is inlined
//!   here (`markov_step_bias` / `sample_block`). When the engine's `hanzo-ml`
//!   pin includes `markov_head`, delegate to it and delete the local helpers.

use hanzo_ml::{DType, Device, Module, Result, Tensor, D};
use hanzo_nn::{Embedding, Linear};
use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
use serde::Deserialize;
use std::sync::Arc;

use crate::layers::{embedding, linear, linear_no_bias, RmsNorm, RotaryEmbedding};
use crate::speculative::{
    SpeculativeProposal, SpeculativeProposalBatch, SpeculativeProposeBatchCtx, SpeculativeProposer,
    TargetTokenEmbedder,
};

#[derive(Debug, Clone, Deserialize)]
pub struct RopeParameters {
    pub rope_theta: f64,
}

/// Draft config, parsed from the DSpark checkpoint `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DSparkConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub rope_parameters: RopeParameters,
    pub block_size: usize,
    pub mask_token_id: u32,
    pub markov_rank: usize,
    pub target_layer_ids: Vec<i64>,
    #[serde(default)]
    pub enable_confidence_head: bool,
    #[serde(default)]
    pub confidence_head_with_markov: bool,
}

impl DSparkConfig {
    pub fn from_json_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let s = std::fs::read_to_string(path).map_err(|e| hanzo_ml::Error::msg(e.to_string()))?;
        serde_json::from_str(&s).map_err(|e| hanzo_ml::Error::msg(e.to_string()))
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    /// Number of fused target-layer hidden states (== `fc` input factor).
    pub fn num_fused_layers(&self) -> usize {
        self.target_layer_ids.len()
    }
}

/// One DSpark decoder layer: a standard Qwen3 layer whose attention keys/values
/// come from `[fused_context ‖ block]` instead of a growing self-attention cache.
struct DSparkLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DSparkLayer {
    fn load(cfg: &DSparkConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let hd = cfg.head_dim;
        let q_dim = cfg.num_attention_heads * hd;
        let kv_dim = cfg.num_key_value_heads * hd;
        let eps = cfg.rms_norm_eps;

        let va = vb.pp("self_attn");
        Ok(Self {
            q_proj: linear_no_bias(h, q_dim, va.pp("q_proj"))?,
            k_proj: linear_no_bias(h, kv_dim, va.pp("k_proj"))?,
            v_proj: linear_no_bias(h, kv_dim, va.pp("v_proj"))?,
            o_proj: linear_no_bias(q_dim, h, va.pp("o_proj"))?,
            q_norm: RmsNorm::new(hd, eps, va.pp("q_norm"))?,
            k_norm: RmsNorm::new(hd, eps, va.pp("k_norm"))?,
            input_layernorm: RmsNorm::new(h, eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(h, eps, vb.pp("post_attention_layernorm"))?,
            gate_proj: linear_no_bias(h, cfg.intermediate_size, vb.pp("mlp").pp("gate_proj"))?,
            up_proj: linear_no_bias(h, cfg.intermediate_size, vb.pp("mlp").pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, h, vb.pp("mlp").pp("down_proj"))?,
        })
    }
}

/// Native Qwen3 DSpark draft model.
pub struct Qwen3DSpark {
    embed_tokens: Embedding,
    fc: Linear,
    hidden_norm: RmsNorm,
    layers: Vec<DSparkLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    markov_w1: Tensor, // [vocab, rank] — nn.Embedding weight
    markov_w2: Tensor, // [vocab, rank] — nn.Linear(rank -> vocab) weight
    confidence_head: Option<Linear>, // Linear(hidden + rank -> 1)
    rotary: RotaryEmbedding,
    cfg: DSparkConfig,
    device: Device,
    dtype: DType,
}

impl Qwen3DSpark {
    pub fn config(&self) -> &DSparkConfig {
        &self.cfg
    }

    /// Load every DSpark tensor from a root VarBuilder (no `model.` prefix).
    pub fn load(cfg: DSparkConfig, vb: ShardedVarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let device = vb.device().clone();
        let dtype = vb.dtype();

        let embed_tokens = embedding(cfg.vocab_size, h, vb.pp("embed_tokens"), &None)?;
        let fc = linear_no_bias(cfg.num_fused_layers() * h, h, vb.pp("fc"))?;
        let hidden_norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("hidden_norm"))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(DSparkLayer::load(&cfg, vb.pp("layers").pp(i))?);
        }

        let norm = RmsNorm::new(h, cfg.rms_norm_eps, vb.pp("norm"))?;
        let lm_head = linear_no_bias(h, cfg.vocab_size, vb.pp("lm_head"))?;

        let markov_w1 = vb
            .pp("markov_head")
            .pp("markov_w1")
            .get((cfg.vocab_size, cfg.markov_rank), "weight")?;
        let markov_w2 = vb
            .pp("markov_head")
            .pp("markov_w2")
            .get((cfg.vocab_size, cfg.markov_rank), "weight")?;

        let confidence_head = if cfg.enable_confidence_head {
            let in_dim = if cfg.confidence_head_with_markov {
                h + cfg.markov_rank
            } else {
                h
            };
            Some(linear(in_dim, 1, vb.pp("confidence_head").pp("proj"))?)
        } else {
            None
        };

        let rotary = RotaryEmbedding::new(
            cfg.rope_theta() as f32,
            cfg.head_dim,
            cfg.max_position_embeddings,
            &device,
            true, // gpt-neox / rotate_half convention (Qwen3)
            dtype,
        )?;

        Ok(Self {
            embed_tokens,
            fc,
            hidden_norm,
            layers,
            norm,
            lm_head,
            markov_w1,
            markov_w2,
            confidence_head,
            rotary,
            cfg,
            device,
            dtype,
        })
    }

    /// Draft one block of `block_size` tokens.
    ///
    /// * `target_hiddens` — the `num_fused_layers` target decoder-layer hidden
    ///   states (layers 1,9,17,25,33), each `[ctx_len, hidden]` (a leading batch
    ///   dim of 1 is accepted and squeezed).
    /// * `anchor_token` — the known token seeding block slot 0.
    /// * `anchor_pos` — RoPE position of the anchor; draft slot `i` sits at
    ///   `anchor_pos + i`, context spans positions `0..ctx_len`.
    /// * `temperature` — Markov block sampling temperature (`< 1e-5` ⇒ argmax).
    ///
    /// Returns `(tokens [block_size] u32, corrected_logits [block_size, vocab],
    /// confidence [block_size])`.
    pub fn draft_block(
        &self,
        target_hiddens: &[Tensor],
        anchor_token: u32,
        anchor_pos: usize,
        temperature: f64,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let dev = &self.device;
        let bs = self.cfg.block_size;

        if target_hiddens.len() != self.cfg.num_fused_layers() {
            return Err(hanzo_ml::Error::msg(format!(
                "draft_block expected {} target hidden states, got {}",
                self.cfg.num_fused_layers(),
                target_hiddens.len()
            )));
        }

        // 1. extract_context_feature: concat the fused target hiddens along the
        //    feature dim, then fuse (fc) and normalize (hidden_norm).
        let mut ctx_parts = Vec::with_capacity(target_hiddens.len());
        for t in target_hiddens {
            ctx_parts.push(as_2d(t)?.to_dtype(self.dtype)?);
        }
        let ctx_refs: Vec<&Tensor> = ctx_parts.iter().collect();
        let ctx_cat = Tensor::cat(&ctx_refs, D::Minus1)?; // [ctx_len, num_fused * hidden]
        let ctx_len = ctx_cat.dim(0)?;
        let tctx = self.hidden_norm.forward(&self.fc.forward(&ctx_cat)?)?; // [ctx_len, hidden]

        // 2. create_noise_embed: [anchor, mask, mask, ...] for the block.
        let mut ids = Vec::with_capacity(bs);
        ids.push(anchor_token);
        ids.extend(std::iter::repeat(self.cfg.mask_token_id).take(bs - 1));
        let ids_t = Tensor::from_vec(ids, (bs,), dev)?;
        let mut hstate = self.embed_tokens.forward(&ids_t)?.to_dtype(self.dtype)?; // [bs, hidden]

        // 3. RoPE tables (draft positions for q, full context+draft positions for k).
        let draft_pos: Vec<u32> = (0..bs as u32).map(|i| anchor_pos as u32 + i).collect();
        let mut full_pos: Vec<u32> = (0..ctx_len as u32).collect();
        full_pos.extend_from_slice(&draft_pos);
        let draft_pos_t = Tensor::from_vec(draft_pos, (bs,), dev)?;
        let full_pos_t = Tensor::from_vec(full_pos, (ctx_len + bs,), dev)?;
        let (cos_draft, sin_draft) = self.rope_cos_sin(&draft_pos_t)?;
        let (cos_full, sin_full) = self.rope_cos_sin(&full_pos_t)?;

        // 4. DSpark attention mask: draft slots see context positions < anchor_pos
        //    and all block slots (bidirectional within the block).
        let mask = self.dspark_mask(ctx_len, bs, anchor_pos)?;

        // 5. backbone.
        for layer in &self.layers {
            let normed = layer.input_layernorm.forward(&hstate)?;
            let attn = self.attention(
                layer, &normed, &tctx, &cos_draft, &sin_draft, &cos_full, &sin_full, &mask,
            )?;
            hstate = hstate.add(&attn)?;
            let normed2 = layer.post_attention_layernorm.forward(&hstate)?;
            let gate = hanzo_nn::ops::silu(&layer.gate_proj.forward(&normed2)?)?;
            let up = layer.up_proj.forward(&normed2)?;
            let mlp = layer.down_proj.forward(&gate.mul(&up)?)?;
            hstate = hstate.add(&mlp)?;
        }
        let hstate = self.norm.forward(&hstate)?; // [bs, hidden]

        // 6. lm_head -> parallel per-slot base logits.
        let base_logits = self.lm_head.forward(&hstate)?; // [bs, vocab]

        // 7. vanilla Markov block sampling (mirror of hanzo_ml::markov_head).
        let (tokens, corrected) = self.sample_block(&base_logits, anchor_token, temperature)?;

        // 8. confidence head on [hidden ‖ W1[prev]] per slot.
        let confidence = self.predict_confidence(&hstate, anchor_token, &tokens)?;

        let tokens_t = Tensor::from_vec(tokens, (bs,), dev)?;
        Ok((tokens_t, corrected, confidence))
    }

    // --- internals -------------------------------------------------------

    /// DSpark attention for one layer. `x`: block hidden `[bs, hidden]`;
    /// `tctx`: fused context `[ctx_len, hidden]`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        layer: &DSparkLayer,
        x: &Tensor,
        tctx: &Tensor,
        cos_draft: &Tensor,
        sin_draft: &Tensor,
        cos_full: &Tensor,
        sin_full: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let bs = x.dim(0)?;
        let ctx_len = tctx.dim(0)?;
        let kl = ctx_len + bs;
        let hd = self.cfg.head_dim;
        let nh = self.cfg.num_attention_heads;
        let nkv = self.cfg.num_key_value_heads;
        let nrep = nh / nkv;
        let eps = self.cfg.rms_norm_eps as f32;

        // Query: block only.
        let q = layer.q_proj.forward(x)?.reshape((bs, nh, hd))?;
        let q = hanzo_nn::ops::rms_norm(&q.contiguous()?, layer.q_norm.weight(), eps)?;
        let q = q.transpose(0, 1)?.contiguous()?; // [nh, bs, hd]
        let q = apply_rope(&q, cos_draft, sin_draft)?;

        // Keys/values: [fused context ‖ block].
        let k_ctx = layer.k_proj.forward(tctx)?;
        let k_noise = layer.k_proj.forward(x)?;
        let k = Tensor::cat(&[&k_ctx, &k_noise], 0)?.reshape((kl, nkv, hd))?;
        let k = hanzo_nn::ops::rms_norm(&k.contiguous()?, layer.k_norm.weight(), eps)?;
        let k = k.transpose(0, 1)?.contiguous()?; // [nkv, kl, hd]
        let k = apply_rope(&k, cos_full, sin_full)?;
        let k = repeat_kv(&k, nrep)?; // [nh, kl, hd]

        let v_ctx = layer.v_proj.forward(tctx)?;
        let v_noise = layer.v_proj.forward(x)?;
        let v = Tensor::cat(&[&v_ctx, &v_noise], 0)?
            .reshape((kl, nkv, hd))?
            .transpose(0, 1)?
            .contiguous()?; // [nkv, kl, hd]
        let v = repeat_kv(&v, nrep)?; // [nh, kl, hd]

        let scale = 1.0 / (hd as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(1, 2)?.contiguous()?)? // [nh, bs, kl]
            .affine(scale, 0.0)?;
        let scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;
        let probs = hanzo_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v.contiguous()?)?; // [nh, bs, hd]
        let out = out.transpose(0, 1)?.contiguous()?.reshape((bs, nh * hd))?;
        layer.o_proj.forward(&out)
    }

    /// Gather + expand the neox RoPE cos/sin rows for the given positions.
    fn rope_cos_sin(&self, positions: &Tensor) -> Result<(Tensor, Tensor)> {
        let (cos, sin) = self.rotary.get_cos_sin()?; // [max_pos, hd/2]
        let cos = cos.index_select(positions, 0)?; // [L, hd/2]
        let sin = sin.index_select(positions, 0)?;
        let cos = Tensor::cat(&[&cos, &cos], D::Minus1)?; // [L, hd]
        let sin = Tensor::cat(&[&sin, &sin], D::Minus1)?;
        Ok((cos, sin))
    }

    /// Additive attention mask `[bs, ctx_len + bs]`: `0` where attended, `-inf`
    /// where masked. Context columns are attended iff `col < anchor_pos`; every
    /// block column is attended (bidirectional block self-attention).
    fn dspark_mask(&self, ctx_len: usize, bs: usize, anchor_pos: usize) -> Result<Tensor> {
        let kl = ctx_len + bs;
        let neg = f32::NEG_INFINITY;
        let mut row = vec![0f32; kl];
        for (j, cell) in row.iter_mut().enumerate().take(ctx_len) {
            if j >= anchor_pos {
                *cell = neg;
            }
        }
        let mut data = Vec::with_capacity(bs * kl);
        for _ in 0..bs {
            data.extend_from_slice(&row);
        }
        Tensor::from_vec(data, (bs, kl), &self.device)?.to_dtype(self.dtype)
    }

    /// Vanilla Markov bias for a single previous token: `W2 · W1[prev]` → `[1, vocab]`.
    /// Mirrors `hanzo_ml::markov_head::vanilla_step_bias`.
    fn markov_step_bias(&self, prev: u32) -> Result<Tensor> {
        let idx = Tensor::from_vec(vec![prev], (1,), &self.device)?;
        let latent = self.markov_w1.index_select(&idx, 0)?; // [1, rank]
        latent.matmul(&self.markov_w2.t()?.contiguous()?) // [1, vocab]
    }

    /// Autoregressive vanilla-Markov block sampling. Mirrors
    /// `hanzo_ml::markov_head::vanilla_sample_block`: slot `k`'s sampled token
    /// becomes slot `k+1`'s previous token. Returns `(tokens, corrected [bs, vocab])`.
    fn sample_block(
        &self,
        base_logits: &Tensor,
        first_prev: u32,
        temperature: f64,
    ) -> Result<(Vec<u32>, Tensor)> {
        let bs = base_logits.dim(0)?;
        let mut tokens = Vec::with_capacity(bs);
        let mut rows = Vec::with_capacity(bs);
        let mut prev = first_prev;
        for k in 0..bs {
            let logits_k = base_logits.narrow(0, k, 1)?; // [1, vocab]
            let corrected_k = logits_k.add(&self.markov_step_bias(prev)?)?; // [1, vocab]
            let tok = select_token(&corrected_k, temperature)?;
            rows.push(corrected_k);
            tokens.push(tok);
            prev = tok;
        }
        let row_refs: Vec<&Tensor> = rows.iter().collect();
        let corrected = Tensor::cat(&row_refs, 0)?; // [bs, vocab]
        Ok((tokens, corrected))
    }

    /// Per-slot accept-rate prediction. Features = `[hidden ‖ W1[prev]]` where
    /// `prev` at slot `k` is `anchor` (k=0) or the token sampled at slot `k-1`.
    fn predict_confidence(
        &self,
        hstate: &Tensor,
        anchor_token: u32,
        tokens: &[u32],
    ) -> Result<Tensor> {
        let bs = hstate.dim(0)?;
        let Some(conf) = &self.confidence_head else {
            return Tensor::zeros((bs,), self.dtype, &self.device);
        };
        let features = if self.cfg.confidence_head_with_markov {
            let mut prev = Vec::with_capacity(bs);
            prev.push(anchor_token);
            prev.extend_from_slice(&tokens[..bs - 1]);
            let prev_t = Tensor::from_vec(prev, (bs,), &self.device)?;
            let prev_emb = self.markov_w1.index_select(&prev_t, 0)?; // [bs, rank]
            Tensor::cat(&[hstate, &prev_emb], D::Minus1)? // [bs, hidden + rank]
        } else {
            hstate.clone()
        };
        conf.forward(&features)?.squeeze(D::Minus1) // [bs]
    }
}

/// Squeeze an optional leading batch dim: `[1, ctx, h]` or `[ctx, h]` → `[ctx, h]`.
fn as_2d(t: &Tensor) -> Result<Tensor> {
    match t.rank() {
        2 => Ok(t.clone()),
        3 => {
            let (b, _, _) = t.dims3()?;
            if b != 1 {
                return Err(hanzo_ml::Error::msg(format!(
                    "target hidden batch dim must be 1, got {b}"
                )));
            }
            t.squeeze(0)
        }
        r => Err(hanzo_ml::Error::msg(format!(
            "target hidden must be rank 2 or 3, got rank {r}"
        ))),
    }
}

/// neox `rotate_half`: split the last dim in half → `[-x2, x1]`.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, d / 2)?;
    let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply neox RoPE. `x`: `[heads, L, head_dim]`; `cos`/`sin`: `[L, head_dim]`.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let cos = cos.unsqueeze(0)?; // [1, L, head_dim]
    let sin = sin.unsqueeze(0)?;
    let rot = rotate_half(x)?;
    x.broadcast_mul(&cos)?.add(&rot.broadcast_mul(&sin)?)
}

/// GQA key/value expansion (repeat_interleave): `[nkv, L, d]` → `[nkv*nrep, L, d]`.
fn repeat_kv(x: &Tensor, nrep: usize) -> Result<Tensor> {
    if nrep == 1 {
        return Ok(x.clone());
    }
    let (nkv, l, d) = x.dims3()?;
    x.unsqueeze(1)?
        .broadcast_as((nkv, nrep, l, d))?
        .contiguous()?
        .reshape((nkv * nrep, l, d))
}

/// Token selection mirroring `hanzo_ml::markov_head::select_token`:
/// `temperature < 1e-5` ⇒ argmax; else multinomial over `softmax(logits/temp)`.
fn select_token(logits: &Tensor, temperature: f64) -> Result<u32> {
    if temperature < 1e-5 {
        let idx = logits.argmax(D::Minus1)?; // [1] u32
        return Ok(idx.to_vec1::<u32>()?[0]);
    }
    let scaled = logits.affine(1.0 / temperature, 0.0)?;
    let row = scaled.to_dtype(DType::F32)?.to_vec2::<f32>()?[0].clone();
    let mx = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = row.iter().map(|v| (v - mx).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    use rand::Rng;
    let r: f32 = rand::rng().random::<f32>();
    let mut acc = 0f32;
    let mut idx = probs.len() - 1;
    for (i, p) in probs.iter().enumerate() {
        acc += *p;
        if r <= acc {
            idx = i;
            break;
        }
    }
    Ok(idx as u32)
}

/// Build a root VarBuilder over one DSpark safetensors shard.
pub fn dspark_varbuilder(
    path: &std::path::Path,
    dtype: DType,
    device: &Device,
) -> Result<ShardedVarBuilder> {
    let predicate: Arc<dyn Fn(String) -> bool + Send + Sync> = Arc::new(|_| true);
    unsafe { ShardedSafeTensors::sharded(&[path], dtype, device, None, predicate) }
}

/// The `SpeculativeProposer` adapter plugging DSpark into the generic speculative driver.
///
/// DSpark is a parallel-block draft: one forward through the [`Qwen3DSpark`] draft model
/// produces a whole `block_size` block of candidate tokens from the target's multi-layer
/// hidden prefix (`ctx.target_hidden_layers`, one `[prefix_len, hidden]` tensor per fused
/// target layer) plus the just-sampled anchor. Correctness comes from the target verify
/// forward, NOT from the draft, so the confidence gate below only trades draft depth.
pub struct DsparkProposer {
    draft: Qwen3DSpark,
    /// Keep only the leading run of draft slots whose `sigmoid(confidence) >= threshold`
    /// (mirrors `_confident_prefix_length` in deepspec/eval/dspark/draft_ops.py). `0.0`
    /// keeps the whole block (default). Never affects correctness — the target verify
    /// decides every emitted token; a shorter draft just narrows the verify window.
    confidence_threshold: f32,
}

impl DsparkProposer {
    pub fn new(draft: Qwen3DSpark, confidence_threshold: f32) -> Self {
        Self {
            draft,
            confidence_threshold,
        }
    }
}

/// Length of the leading confident prefix. Mirrors DeepSpec's `_confident_prefix_length`:
/// walk the per-slot confidences, stop at the first slot whose `sigmoid(conf)` drops below
/// the threshold, and always keep at least one token. `threshold <= 0` ⇒ the whole block.
/// A pure function so the gate is testable without loading a draft model.
fn confident_prefix_length(threshold: f32, confidence: &[f32], block: usize) -> usize {
    if threshold <= 0.0 {
        return block;
    }
    let mut n = 0usize;
    for &c in confidence {
        let p = 1.0f32 / (1.0 + (-c).exp());
        if p >= threshold {
            n += 1;
        } else {
            break;
        }
    }
    n.max(1)
}

impl SpeculativeProposer for DsparkProposer {
    fn proposal_len(&self) -> usize {
        self.draft.config().block_size
    }

    fn propose(
        &mut self,
        ctx: SpeculativeProposeBatchCtx<'_>,
        _target_embedder: Option<&TargetTokenEmbedder<'_>>,
    ) -> Result<SpeculativeProposalBatch> {
        let hiddens = ctx.target_hidden_layers.as_ref().ok_or_else(|| {
            hanzo_ml::Error::Msg(
                "DSpark proposer requires the multi-layer target hidden prefix".into(),
            )
        })?;
        let batch = ctx.sampled_tokens.len();
        if batch == 0 {
            return Ok(SpeculativeProposalBatch::new(Vec::new()));
        }
        // DSpark drafts from ONE sequence's full prefix context (the captured hiddens are that
        // sequence's forward). Batched multi-sequence drafting needs a per-sequence prefix
        // slice and is a follow-on; the supported shape here is a single active sequence.
        if batch != 1 {
            hanzo_ml::bail!(
                "DSpark proposer drafts one sequence per step (got batch={batch})"
            );
        }
        let anchor_token = ctx.sampled_tokens[0];
        let anchor_pos = ctx.base_lens[0];
        let block = self.draft.config().block_size;

        // Gap 2: DSpark drafts against the fused hiddens of the WHOLE confirmed prefix. The target
        // model hands us its accumulated confirmed-prefix buffer (`[prefix_len, hidden]` per fused
        // layer); slice it to the `anchor_pos` positions that precede the anchor. In the steady
        // state `prefix_len == anchor_pos`; a shorter prefix means the accumulator hasn't caught up
        // (e.g. right after a discontinuity) — bail so the target simply decodes one token.
        let prefix_len = hiddens.first().map(|t| t.dim(0)).transpose()?.unwrap_or(0);
        if prefix_len < anchor_pos {
            return Ok(SpeculativeProposalBatch::new(vec![SpeculativeProposal::new(
                Vec::new(),
            )]));
        }
        let prefix = hiddens
            .iter()
            .map(|t| t.narrow(0, 0, anchor_pos))
            .collect::<Result<Vec<_>>>()?;

        // Deterministic draft (argmax): draft quality only affects accept rate, and the
        // target verify decides every emitted token.
        let (tokens_t, logits, confidence) =
            self.draft.draft_block(&prefix, anchor_token, anchor_pos, 0.0)?;

        let tokens: Vec<u32> = tokens_t.to_vec1::<u32>()?;
        let conf: Vec<f32> = confidence.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let keep =
            confident_prefix_length(self.confidence_threshold, &conf, block).min(tokens.len());

        // Truncate to the confident prefix. The verifier indexes logit rows by draft
        // position and expects `[1, keep, vocab]`.
        let tokens = tokens[..keep].to_vec();
        let logits = logits.narrow(0, 0, keep)?.unsqueeze(0)?; // [1, keep, vocab]

        Ok(SpeculativeProposalBatch::new(vec![
            SpeculativeProposal::with_logits(tokens, logits),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CKPT_DIR: &str = "/home/z/work/zen/hf/dspark/dspark_qwen3_4b_block7";

    /// Checkpoint override (env DSPARK_CKPT) so the load tests can point at ANY
    /// DSpark checkpoint — e.g. one produced by the native hanzo-train trainer
    /// (the deploy arrow of the all-Rust dump→cache→train→deploy loop).
    fn ckpt_dir() -> String {
        std::env::var("DSPARK_CKPT").unwrap_or_else(|_| CKPT_DIR.to_string())
    }

    /// Pure gate logic — runs on CPU with no model. `threshold <= 0` keeps the whole block;
    /// otherwise keep the leading run whose `sigmoid(conf) >= threshold`, always at least one.
    #[test]
    fn dspark_confident_prefix_length_logic() {
        // sigmoid(+10) ~ 1.0 (confident), sigmoid(-10) ~ 0.0 (not).
        let hi = 10.0f32;
        let lo = -10.0f32;
        // threshold 0 => full block regardless of confidences.
        assert_eq!(confident_prefix_length(0.0, &[lo, lo, lo], 7), 7);
        // all confident => keep all provided slots.
        assert_eq!(confident_prefix_length(0.5, &[hi, hi, hi], 3), 3);
        // stop at first below-threshold slot.
        assert_eq!(confident_prefix_length(0.5, &[hi, hi, lo, hi], 4), 2);
        // never zero: even a first low slot keeps one.
        assert_eq!(confident_prefix_length(0.5, &[lo, hi, hi], 3), 1);
    }

    /// End-to-end proposer over the real checkpoint IF present (skips gracefully otherwise):
    /// 5 random `[12, hidden]` target hiddens + anchor -> a `1..=block` token proposal whose
    /// logits are `[1, keep, vocab]`. No GPU, no model download.
    #[test]
    fn dspark_proposer_produces_block() -> Result<()> {
        use crate::speculative::SpeculativeKvCache;
        use rand::SeedableRng;
        use rand_isaac::Isaac64Rng;
        use std::sync::{Arc, Mutex};

        let dir_s = ckpt_dir();
        let dir = std::path::Path::new(&dir_s);
        let weights = dir.join("model.safetensors");
        if !weights.exists() {
            eprintln!("skipping: checkpoint not found at {}", weights.display());
            return Ok(());
        }

        let dev = Device::Cpu;
        let cfg = DSparkConfig::from_json_file(dir.join("config.json"))?;
        let block = cfg.block_size;
        let n_fused = cfg.num_fused_layers();
        let h = cfg.hidden_size;
        let vocab = cfg.vocab_size;
        let vb = dspark_varbuilder(&weights, DType::F32, &dev)?;
        let draft = Qwen3DSpark::load(cfg, vb)?;
        let mut proposer = DsparkProposer::new(draft, 0.0);
        assert_eq!(proposer.proposal_len(), block);

        // 5 target-layer prefix hiddens [ctx_len, hidden].
        let ctx_len = 12usize;
        let mut hiddens = Vec::with_capacity(n_fused);
        for _ in 0..n_fused {
            hiddens.push(Tensor::randn(0f32, 1f32, (ctx_len, h), &dev)?);
        }

        let anchor_token = 12345u32;
        let sampled = [anchor_token];
        let base_lens = [ctx_len]; // anchor_pos = prefix length
        let seq_ids = [0usize];
        let rng = Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0)));
        // DSpark's propose reads only sampled_tokens / base_lens / target_hidden_layers, so an
        // empty `sequences` slice and a Normal cache stub are sufficient for a unit test.
        let ctx = SpeculativeProposeBatchCtx {
            sampled_tokens: &sampled,
            sampled_tokens_emitted: true,
            seq_ids: &seq_ids,
            base_lens: &base_lens,
            sequences: &[],
            cache: SpeculativeKvCache::Normal,
            target_hiddens: None,
            target_hidden_layers: Some(hiddens),
            rng,
        };

        let batch = proposer.propose(ctx, None)?;
        assert_eq!(batch.proposals.len(), 1);
        let p = &batch.proposals[0];
        assert!(
            (1..=block).contains(&p.tokens.len()),
            "proposal len {} out of 1..={block}",
            p.tokens.len()
        );
        let logits = p
            .logits
            .as_ref()
            .expect("DSpark proposal must carry logits");
        assert_eq!(logits.dims(), &[1, p.tokens.len(), vocab]);
        assert!(
            p.tokens.iter().all(|&t| (t as usize) < vocab),
            "draft token out of vocab range"
        );
        eprintln!("dspark proposer tokens = {:?}", p.tokens);
        Ok(())
    }

    #[test]
    fn dspark_load_and_draft_block() -> Result<()> {
        let dir_s = ckpt_dir();
        let dir = std::path::Path::new(&dir_s);
        let weights = dir.join("model.safetensors");
        if !weights.exists() {
            eprintln!("skipping: checkpoint not found at {}", weights.display());
            return Ok(());
        }

        let dev = Device::Cpu;
        let cfg = DSparkConfig::from_json_file(dir.join("config.json"))?;
        assert_eq!(cfg.block_size, 7);
        if std::env::var("DSPARK_CKPT").is_err() {
            // Reference-checkpoint invariants (deepseek-ai/dspark_qwen3_4b_block7).
            // Overridden checkpoints (e.g. hanzo-train output) carry their own dims
            // in config.json — the load below validates shape consistency itself.
            assert_eq!(cfg.num_hidden_layers, 5);
            assert_eq!(cfg.num_fused_layers(), 5);
            assert_eq!(cfg.mask_token_id, 151669);
        }
        let n_fused = cfg.num_fused_layers();

        let vb = dspark_varbuilder(&weights, DType::F32, &dev)?;
        let model = Qwen3DSpark::load(cfg, vb)?;

        // Random target hiddens: n_fused × [ctx_len, hidden].
        let ctx_len = 6usize;
        let h = model.config().hidden_size;
        let vocab = model.config().vocab_size;
        let mut hiddens = Vec::with_capacity(n_fused);
        for _ in 0..n_fused {
            hiddens.push(Tensor::randn(0f32, 1f32, (ctx_len, h), &dev)?);
        }

        let anchor_token = 12345u32;
        let anchor_pos = ctx_len - 1;
        let (tokens, logits, confidence) =
            model.draft_block(&hiddens, anchor_token, anchor_pos, 0.0)?;

        // Structural asserts.
        assert_eq!(tokens.dims(), &[7]);
        assert_eq!(logits.dims(), &[7, vocab]);
        assert_eq!(confidence.dims(), &[7]);

        // Numeric sanity: finite everywhere, tokens in range.
        let logit_vals = logits.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            logit_vals.iter().all(|v| v.is_finite()),
            "block logits contain non-finite values"
        );
        let conf_vals = confidence.to_vec1::<f32>()?;
        assert!(
            conf_vals.iter().all(|v| v.is_finite()),
            "confidence contains non-finite values"
        );
        let toks = tokens.to_vec1::<u32>()?;
        assert!(
            toks.iter().all(|&t| (t as usize) < vocab),
            "draft token out of vocab range"
        );

        eprintln!("draft tokens   = {toks:?}");
        eprintln!("confidence     = {conf_vals:?}");
        eprintln!(
            "logits[0][..5] = {:?}",
            &logits.narrow(0, 0, 1)?.flatten_all()?.to_vec1::<f32>()?[..5]
        );
        Ok(())
    }

    /// Cross-language parity: the Rust `draft_block` must reproduce the DeepSpec
    /// Python reference (deepspec/eval/dspark/draft_ops.py) bit-close on identical
    /// inputs. The reference dump is produced by `scratchpad/dspark_ref_dump.py`
    /// (fixed seed, F32): 5 target-layer hiddens [ctx_len,H] + corrected block
    /// logits [bs,vocab] + tokens [bs] + confidence [bs], with ctx_len=12,
    /// anchor_pos=12, anchor_token=100. Run explicitly:
    ///   cargo test -p hanzo-engine dspark_matches_python_reference -- --ignored --nocapture
    #[test]
    #[ignore = "needs the Python reference dump + local checkpoint"]
    fn dspark_matches_python_reference() -> Result<()> {
        const REF: &str = "/tmp/claude-1000/-home-z-work-lux/\
95715740-b8bb-4d96-8a9c-010a600ec9a6/scratchpad/dspark_ref.safetensors";
        // Must match dspark_ref_dump.py.
        const CTX_LEN: usize = 12;
        const ANCHOR_POS: usize = 12;
        const ANCHOR_TOKEN: u32 = 100;

        let dir_s = ckpt_dir();
        let dir = std::path::Path::new(&dir_s);
        let weights = dir.join("model.safetensors");
        let refp = std::path::Path::new(REF);
        if !weights.exists() || !refp.exists() {
            eprintln!("skipping: need checkpoint {} + ref dump {}", weights.display(), REF);
            return Ok(());
        }

        let dev = Device::Cpu;
        let cfg = DSparkConfig::from_json_file(dir.join("config.json"))?;
        let vb = dspark_varbuilder(&weights, DType::F32, &dev)?;
        let model = Qwen3DSpark::load(cfg, vb)?;

        // Load the reference dump (same inputs the Python side used).
        let refs = hanzo_ml::safetensors::load(refp, &dev)?;
        let hiddens: Vec<Tensor> = (0..model.config().num_fused_layers())
            .map(|i| refs[&format!("target_hidden_{i}")].to_dtype(DType::F32))
            .collect::<Result<_>>()?;
        assert_eq!(hiddens[0].dims(), &[CTX_LEN, model.config().hidden_size]);

        let (tokens, logits, confidence) =
            model.draft_block(&hiddens, ANCHOR_TOKEN, ANCHOR_POS, 0.0)?;

        // Compare the Markov-corrected block logits (backbone + Markov) — the
        // decisive numeric check. Small tolerance for F32 reduction-order drift.
        let got = logits.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let exp = refs["corrected_logits"].to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got.len(), exp.len(), "logit count mismatch");
        let max_abs = got
            .iter()
            .zip(&exp)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let mean_abs = got.iter().zip(&exp).map(|(a, b)| (a - b).abs()).sum::<f32>()
            / got.len() as f32;
        eprintln!("corrected-logit max|Δ| = {max_abs:.5}, mean|Δ| = {mean_abs:.6}");

        // Token agreement (the operational signal — do the drafts match?).
        let got_toks = tokens.to_vec1::<u32>()?;
        let exp_toks: Vec<u32> = refs["tokens"]
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?;
        eprintln!("rust tokens = {got_toks:?}");
        eprintln!("ref  tokens = {exp_toks:?}");
        let tok_match = got_toks.iter().zip(&exp_toks).filter(|(a, b)| a == b).count();

        // Confidence parity.
        let got_conf = confidence.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let exp_conf = refs["confidence"].to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let conf_max = got_conf
            .iter()
            .zip(&exp_conf)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("rust conf = {got_conf:?}");
        eprintln!("ref  conf = {exp_conf:?}");
        eprintln!("confidence max|Δ| = {conf_max:.5}");

        assert!(
            max_abs < 5e-2,
            "corrected block logits diverge from Python reference (max|Δ|={max_abs})"
        );
        assert_eq!(
            tok_match,
            exp_toks.len(),
            "draft tokens diverge from Python reference: rust {got_toks:?} vs ref {exp_toks:?}"
        );
        assert!(
            conf_max < 5e-2,
            "confidence diverges from Python reference (max|Δ|={conf_max})"
        );
        eprintln!("✓ Rust draft_block matches the DeepSpec Python reference");
        Ok(())
    }
}
