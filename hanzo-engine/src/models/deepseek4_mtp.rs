//! DeepSeek-V4 MTP (Multi-Token-Prediction) speculative draft head —
//! `general.architecture = "deepseek4_mtp_support"` (the separate ~3.6 GB GGUF
//! `DeepSeek-V4-Flash-MTP-…`, `nextn_predict_layers = 1`).
//!
//! The MTP head is structurally ONE V4 decoder block (`mtp.0.*`: attn + MoE + per-
//! sublayer Hyper-Connections — loaded via the shared [`DecoderLayer::load`]) wrapped
//! with a NextN entry/exit: it predicts token `t+1` from the base model's hidden state
//! at `t` plus the embedding of token `t`:
//!
//! ```text
//!   x = e_proj · enorm(embed(token_t))  +  h_proj · hnorm(hidden_t)   // [b,s,e]
//!   carrier = HC.expand(x); carrier = block(carrier); x = head_hc.reduce(carrier)
//!   logits  = base_output( norm(x) )                                   // [b,s,vocab]
//! ```
//!
//! `e_proj`/`h_proj` are two `[e,e]` projections summed (≡ a `[2e,e]` concat-proj),
//! matching DeepSeek's NextN module + llama.cpp's MTP (PR #22673, NextN tensors +
//! `n_layer_nextn`). The output head + token embeddings are SHARED with the base model.
//!
//! Speculative use: research (antirez/ds4 + SGLang) shows naive **depth-1 MTP is
//! net-negative single-stream** (accept ~1.8-2.3, verify 2× → −21%). The value is
//! **EAGLE-style** (accept ~2.5) with **batched verify** (stream weights once for all
//! K draft tokens inside a fixed-shape graph). So [`MtpHead::draft`] is built to be
//! looped for multi-step drafting; the accept/verify loop is the engine-side follow-on.

use std::sync::Arc;

use crate::gguf::Content;
use crate::layers::{
    CausalMaskConfig, CausalMasker, DeepSeekV2RopeConfig, DeepSeekV2RotaryEmbedding, RmsNorm,
};
use crate::layers_masker::PastKvLenCache;
use crate::pipeline::KvCache;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::QuantMethod;

use super::deepseek4::HyperConnections;
use super::quantized_deepseek4::{deq, gguf_linear, rms_from, DecoderLayer, PropsGGUF};

/// The DeepSeek-V4 MTP draft head: a single V4 block + NextN entry/exit. Shares the
/// base model's token embeddings + output projection (passed to [`Self::draft`]).
pub struct MtpHead {
    enorm: RmsNorm,
    hnorm: RmsNorm,
    e_proj: Arc<dyn QuantMethod>,
    h_proj: Arc<dyn QuantMethod>,
    block: DecoderLayer,
    head_hc: HyperConnections,
    norm: RmsNorm,
    n_hc: usize,
    head_dim: usize,
    device: Device,
    dtype: DType,
    /// Single-block KV cache, held as a 1-element Vec so it satisfies `PastKvLenCache`
    /// (mirrors the base model's `NormalCache`).
    cache: Vec<KvCache>,
}

impl MtpHead {
    /// Load the `mtp.0.*` head from an opened MTP-GGUF [`Content`]. `props` is the base
    /// model's config (the MTP block shares all V4 hyperparams). The MTP block is a
    /// Full-mode V4 block (no compressor, bias-routed MoE).
    pub fn load<R: std::io::Seek + std::io::Read>(
        ct: &mut Content<'_, R>,
        props: &PropsGGUF,
        device: &Device,
        dtype: DType,
    ) -> Result<Self> {
        let eps = props.rms_norm_eps as f64;
        let softmax_scale = 1.0 / (props.head_dim as f32).sqrt();
        let group_in = props.head_count * props.head_dim / props.o_groups;

        // Full-mode RoPE (θ=rope_theta, no YaRN) — the MTP block is not compressed.
        let rope = Arc::new(DeepSeekV2RotaryEmbedding::new(
            &DeepSeekV2RopeConfig {
                rope_scaling: None,
                max_position_embeddings: props.max_seq_len,
                rope_theta: props.rope_theta,
                qk_rope_head_dim: props.rope_head_dim,
            },
            DType::F32,
            device,
        )?);

        // The V4 block at `mtp.0` (Full mode, bias-routed, swiglu clamp 10 like V4).
        let block = DecoderLayer::load(
            ct,
            "mtp.0",
            props,
            device,
            rope,
            /*compress_ratio*/ 0,
            group_in,
            eps,
            softmax_scale,
            dtype,
            /*is_hash*/ false,
            /*swiglu_clamp*/ 10.0,
        )?;

        // NextN entry: norm + project the prev hidden and the next-token embedding.
        let enorm = rms_from(deq(ct, "mtp.0.enorm.weight", device)?, eps)?;
        let hnorm = rms_from(deq(ct, "mtp.0.hnorm.weight", device)?, eps)?;
        let e_proj = gguf_linear(ct.tensor("mtp.0.e_proj.weight", device)?)?;
        let h_proj = gguf_linear(ct.tensor("mtp.0.h_proj.weight", device)?)?;
        let norm = rms_from(deq(ct, "mtp.0.norm.weight", device)?, eps)?;

        // Head Hyper-Connection (reduce-only) — `hc_head_*`.
        let head_hc = HyperConnections::from_parts(
            gguf_linear(ct.tensor("mtp.0.hc_head_fn.weight", device)?)?,
            deq(ct, "mtp.0.hc_head_scale.weight", device)?.to_vec1::<f32>()?,
            deq(ct, "mtp.0.hc_head_base.weight", device)?,
            props.hc_count,
            props.hc_sinkhorn_iters,
            props.hc_eps,
            true,
        )?;

        Ok(Self {
            enorm,
            hnorm,
            e_proj,
            h_proj,
            block,
            head_hc,
            norm,
            n_hc: props.hc_count,
            head_dim: props.head_dim,
            device: device.clone(),
            dtype,
            cache: vec![KvCache::new_normal(2, props.max_seq_len, 512)],
        })
    }

    /// Reset the draft head's KV cache (call before each fresh speculative round).
    pub fn reset(&mut self) {
        for c in &mut self.cache {
            c.reset();
        }
    }

    /// Draft the next-token logits from the base model's `hidden` state `[b,s,e]` and
    /// the corresponding `token_ids` `[b,s]`, sharing the base `embed` + `output` head.
    /// `start_offsets` are the per-sequence positions (for RoPE/causality).
    ///
    /// Returns `[b, s, vocab]`. Loop this (feeding back the accepted token's hidden) for
    /// multi-step EAGLE-style drafting.
    pub fn draft(
        &mut self,
        hidden: &Tensor,
        token_ids: &Tensor,
        embed: &Embedding,
        output: &Arc<dyn QuantMethod>,
        start_offsets: &[usize],
    ) -> Result<Tensor> {
        let e = self.e_proj.forward(
            &self
                .enorm
                .forward(&embed.forward(token_ids)?.to_dtype(self.dtype)?)?,
        )?;
        let h = self
            .h_proj
            .forward(&self.hnorm.forward(&hidden.to_dtype(self.dtype)?)?)?;
        let x = (e + h)?; // [b, s, e]

        let mask = CausalMasker.make_causal_mask(
            token_ids,
            &self.cache as &dyn PastKvLenCache,
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let positions = Tensor::from_vec(
            start_offsets.iter().map(|&o| o as u32).collect::<Vec<_>>(),
            start_offsets.len(),
            &self.device,
        )?;

        // One V4 block over the HC carrier (no compressor — Full mode).
        let mut hc = HyperConnections::expand(&x, self.n_hc)?;
        hc = self
            .block
            .forward(&hc, token_ids, &mask, &positions, &mut self.cache[0], None)?;
        let x = self.head_hc.reduce_output(&hc)?;
        let x = self.norm.forward(&x)?;
        output.forward(&x.contiguous()?)
    }

    /// Head dim (for callers wiring the draft into a speculative loop).
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }
}
