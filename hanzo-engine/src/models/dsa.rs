#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! DeepSeek Sparse Attention (DSA) — the shared "lightning indexer" + top-k
//! selection primitive.
//!
//! DSA turns the quadratic MLA attention into a near-linear one. A *cheap*
//! indexer scores every preceding token against the current query; the highest
//! `index_topk` tokens are selected and the (expensive) main attention is
//! restricted to that set. The indexer is trained to approximate the full
//! attention pattern, so applying the selection over the existing
//! (full-precision) MLA key/value state is a correct, faithful implementation
//! — turning `O(L²)` into `O(L·k)` work for the selected attention.
//!
//! This module is the reusable selection primitive shared by
//! DeepSeek-V3.2 / V4 and GLM-5, parameterized purely by
//! `index_n_heads` / `index_head_dim` / `index_topk`.
//!
//! # Indexer math (per the `ds4.c` reference, `indexer_allowed_decode_one`)
//!
//! For a query token `i` and a candidate (key) token `c`:
//!
//! ```text
//!   q[i]      = Wq @ q_src[i]                       -> [n_head, head_dim]   (partial RoPE)
//!   k[c]      = k_norm(Wk @ hidden[c])              -> [head_dim]           (one head, MQA-style)
//!   w[i]      = (Wweight @ hidden[i]) / sqrt(head_dim · n_head)  -> [n_head]
//!   score[i,c] = Σ_h ReLU( dot( k[c], q[i,h] ) ) · w[i,h]
//! ```
//!
//! Note the **ReLU clamp on the dot product only** (never on the weight `w`,
//! which is signed). The key is a *single* `head_dim` vector shared across all
//! query heads (multi-query indexer). The per-head weight `w` is taken from the
//! *query* token's hidden state. The `1/sqrt(head_dim · n_head)` scale is folded
//! into `w`; being a positive global constant it does not change the top-k
//! ordering, only the score magnitude.
//!
//! Selection: `top_k = min(index_topk, n_candidates)` highest scores form a
//! boolean "allowed" mask that gates the main attention.
//!
//! # Full precision vs. FP8 kernel (follow-on)
//!
//! This is the **full-precision** functional implementation: the indexer
//! projections load (and, for real FP8 checkpoints, dequantize via
//! [`hanzo_quant::linear_no_bias`]) to the working dtype, and selection is
//! materialised as an additive `-inf` attention bias over the dense score
//! matrix. The *large* follow-on is the fused **FP8 lightning-indexer kernel**
//! (FP8 q·k with per-head ReLU accumulation) plus the gather-based sparse
//! attention that realises the `O(L·k)` memory win — see [`DsaSelection`].
//!
//! # Activation scope
//!
//! As wired into [`super::deepseek3`], DSA currently runs **only on the eager
//! (`--no-paged-attn`) cold-cache prefill path** (`past_kv_len == 0`), where the
//! selection bias aligns with the `[Lq, Lk]` causal mask. Warm-cache prefill,
//! paged-attention, and MLA-decode fall back to the dense path; fusing DSA into
//! those (and sourcing indexer keys from the full cached K for long-context) is
//! part of the FP8-kernel follow-on.

use std::sync::Arc;

use hanzo_ml::{DType, Device, Module, Result, Tensor, D};
use hanzo_nn::LayerNorm;
use hanzo_quant::{QuantMethod, QuantizedConfig, ShardedVarBuilder};

use crate::layers::{layer_norm, DeepSeekV2RotaryEmbedding};
use crate::ops::TopKLastDimOp;

/// Large finite negative used as the additive `-inf` for masked-out positions.
/// Finite (not `f32::NEG_INFINITY`) so that `0 · bias` can never produce `NaN`
/// and it stays representable after a cast to bf16/f16 (where it saturates to
/// `-inf`, the desired softmax behaviour).
const MASK_NEG_BIAS: f64 = -1e30;

/// Static config for the DSA indexer, parameterized so the same primitive
/// covers DeepSeek-V3.2 / V4 and GLM-5 selection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DsaConfig {
    /// Number of indexer query heads (`index_n_heads`, e.g. 64).
    pub index_n_heads: usize,
    /// Per-head indexer dimension (`index_head_dim`, e.g. 128).
    pub index_head_dim: usize,
    /// Number of preceding tokens to keep per query (`index_topk`, e.g. 2048).
    pub index_topk: usize,
    /// Leading dims of each indexer head that receive (partial) RoPE.
    /// Equals `qk_rope_head_dim` for V3.2; `0` disables RoPE entirely.
    pub rope_dim: usize,
}

/// Result of a DSA selection over a `[batch, q_len, kv_len]` score matrix.
///
/// `Clone` is cheap (both tensors are `Arc`-backed) and lets GLM-5's IndexShare
/// reuse a group-leader layer's selection across the following `"shared"` layers.
#[derive(Clone)]
pub(crate) struct DsaSelection {
    /// Selected key indices per query, `[batch, q_len, min(topk, kv_len)]` (`u32`),
    /// sorted by descending score. Useful for a gather-based sparse attention
    /// (the memory-optimal `O(L·k)` follow-on path).
    #[allow(dead_code)]
    pub indices: Tensor,
    /// Boolean selection mask, `[batch, q_len, kv_len]` (`f32` in `{0, 1}`).
    /// Exactly `min(topk, num_valid_i)` ones per query row `i`, where
    /// `num_valid_i` honours causal masking. This is the authoritative,
    /// causal-corrected selection.
    pub mask: Tensor,
}

impl DsaSelection {
    /// Additive attention bias: `0` where a key is selected, large-negative
    /// (`-inf`) where it is not. Add it (broadcast over heads) to the main
    /// attention's score mask to restrict attention to the selected keys.
    pub fn to_attn_bias(&self, dtype: DType) -> Result<Tensor> {
        // mask 1 -> 0 ; mask 0 -> MASK_NEG_BIAS
        self.mask
            .affine(-MASK_NEG_BIAS, MASK_NEG_BIAS)?
            .to_dtype(dtype)
    }

    /// Fold this selection into a base additive attention-mask tensor (the
    /// causal mask `[Lq, Lk]`), returning a `[B, 1, Lq, Lk]` additive mask
    /// broadcastable over heads: `base + (-inf where the key is not selected)`.
    ///
    /// The selection's `Lk` must equal the base mask's `Lk`. This holds on the
    /// cold-cache prefill path (keys sourced from the current chunk, so
    /// `Lk == Lq`); see the caller's `past_kv_len == 0` guard.
    pub fn combine_with_mask(&self, base: &Tensor) -> Result<Tensor> {
        let bias = self.to_attn_bias(base.dtype())?.unsqueeze(1)?; // [B, 1, Lq, Lk]
        base.broadcast_add(&bias)
    }
}

/// The DSA lightning indexer: cheap per-head q/k projections + a per-head
/// weight projection that together score every candidate token against the
/// query, after which the top-`index_topk` are selected.
pub(crate) struct DsaIndexer {
    /// Query projection `q_src_dim -> index_n_heads · index_head_dim`.
    wq: Arc<dyn QuantMethod>,
    /// Key projection `kv_src_dim -> index_head_dim` (single shared head).
    wk: Arc<dyn QuantMethod>,
    /// Per-head weight projection `kv_src_dim -> index_n_heads`.
    weights_proj: Arc<dyn QuantMethod>,
    /// Optional key normalization (present in V3.2). DeepSeek-V3.2 ships a real
    /// `LayerNorm` here (`indexer.k_norm.{weight,bias}`) — mean-centered with a
    /// learned bias — **not** an RMS norm. Skipped when the tensors are absent.
    k_norm: Option<LayerNorm>,
    cfg: DsaConfig,
    /// `1 / sqrt(index_head_dim · index_n_heads)`, folded into the weights.
    weight_scale: f64,
}

impl DsaIndexer {
    /// Build the indexer from a var-builder rooted at the `indexer.*` tensors.
    ///
    /// Returns `Ok(None)` when the indexer tensors are absent, so a model that
    /// declares the DSA config but ships without indexer weights (or with
    /// differently-named ones) cleanly falls back to the dense path instead of
    /// failing to load. `quant` carries the (optional) quantization config so
    /// real FP8 checkpoints dequantize through [`hanzo_quant::linear_no_bias`].
    pub fn load(
        cfg: DsaConfig,
        q_src_dim: usize,
        kv_src_dim: usize,
        norm_eps: f64,
        quant: &Option<QuantizedConfig>,
        vb: ShardedVarBuilder,
    ) -> Result<Option<Self>> {
        if !vb.contains_tensor("wq_b.weight")
            || !vb.contains_tensor("wk.weight")
            || !vb.contains_tensor("weights_proj.weight")
        {
            return Ok(None);
        }

        let wq = hanzo_quant::linear_no_bias(
            q_src_dim,
            cfg.index_n_heads * cfg.index_head_dim,
            quant,
            vb.pp("wq_b"),
        )?;
        let wk = hanzo_quant::linear_no_bias(kv_src_dim, cfg.index_head_dim, quant, vb.pp("wk"))?;
        let weights_proj = hanzo_quant::linear_no_bias(
            kv_src_dim,
            cfg.index_n_heads,
            quant,
            vb.pp("weights_proj"),
        )?;

        // V3.2's indexer key norm is a real LayerNorm (mean-centered, learned
        // weight + bias), loaded via `layer_norm` (affine + remove_mean) — using
        // RmsNorm here would silently drop the bias and skip mean-centering,
        // yielding wrong key vectors and wrong top-k.
        let k_norm = if vb.contains_tensor("k_norm.weight") {
            Some(layer_norm(cfg.index_head_dim, norm_eps, vb.pp("k_norm"))?)
        } else {
            None
        };

        Ok(Some(Self::from_parts(cfg, wq, wk, weights_proj, k_norm)))
    }

    /// Assemble an indexer from already-built projections — the source-agnostic
    /// constructor. [`load`](Self::load) builds `wq`/`wk`/`weights_proj` from a
    /// safetensors var-builder; the GGUF quantized path
    /// ([`super::quantized_deepseek2`]) builds the same three from GGUF
    /// `GgufMatMul` tensors. Both converge here so the folded `weight_scale` and
    /// the [`forward`](Self::forward) math live in exactly one place, independent
    /// of where the weights came from.
    pub(crate) fn from_parts(
        cfg: DsaConfig,
        wq: Arc<dyn QuantMethod>,
        wk: Arc<dyn QuantMethod>,
        weights_proj: Arc<dyn QuantMethod>,
        k_norm: Option<LayerNorm>,
    ) -> Self {
        let weight_scale = 1.0 / ((cfg.index_head_dim * cfg.index_n_heads) as f64).sqrt();
        Self {
            wq,
            wk,
            weights_proj,
            k_norm,
            cfg,
            weight_scale,
        }
    }

    /// Run the indexer over a self-attention block and return the DSA selection.
    ///
    /// `q_src` (`[B, L, q_src_dim]`) feeds the query projection (the
    /// q-LoRA-normalised latent in V3.2); `hidden` (`[B, L, kv_src_dim]`) feeds
    /// both the key projection (per key position) and the weight projection
    /// (per query position). For self-attention these share the sequence length
    /// `L`. RoPE is applied to the indexer query/key when a rotary embedding and
    /// `positions` are supplied and `rope_dim > 0`.
    pub fn forward(
        &self,
        q_src: &Tensor,
        hidden: &Tensor,
        rotary: Option<&DeepSeekV2RotaryEmbedding>,
        positions: Option<&Tensor>,
        causal: bool,
    ) -> Result<DsaSelection> {
        let (b, lq, _) = q_src.dims3()?;
        let nh = self.cfg.index_n_heads;
        let dh = self.cfg.index_head_dim;

        // Queries: [B, Lq, nh·dh] -> [B, nh, Lq, dh]
        let q = self
            .wq
            .forward(q_src)?
            .reshape((b, lq, nh, dh))?
            .transpose(1, 2)?
            .contiguous()?;

        let k = self.keys(hidden)?; // [B, Lk, dh]

        // Per-head, per-query weights with the 1/sqrt(dh·nh) scale folded in.
        let weights = self
            .weights_proj
            .forward(hidden)?
            .affine(self.weight_scale, 0.0)?; // [B, Lq, nh]

        let (q, k) = match (rotary, positions) {
            (Some(rot), Some(pos)) if self.cfg.rope_dim > 0 => {
                apply_partial_rope(&q, &k, self.cfg.rope_dim, rot, pos)?
            }
            _ => (q, k),
        };

        let scores = indexer_scores(&q, &k, &weights)?;
        select_topk(&scores, self.cfg.index_topk, causal)
    }

    /// Project and normalize the indexer keys: `[B, Lk, dh]`, one shared head.
    /// The norm (when present) is a mean-centered LayerNorm with learned bias.
    fn keys(&self, hidden: &Tensor) -> Result<Tensor> {
        let k = self.wk.forward(hidden)?;
        match &self.k_norm {
            Some(k_norm) => k_norm.forward(&k),
            None => Ok(k),
        }
    }
}

/// The DSA score: `score[b,i,c] = Σ_h ReLU( q[b,h,i,:] · k[b,c,:] ) · w[b,i,h]`.
///
/// * `q` — `[B, nh, Lq, dh]` indexer queries.
/// * `k` — `[B, Lk, dh]` indexer keys (single shared head, broadcast over `nh`).
/// * `weights` — `[B, Lq, nh]` per-head query weights (scale pre-folded).
///
/// Returns `[B, Lq, Lk]`.
fn indexer_scores(q: &Tensor, k: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let (b, nh, _lq, dh) = q.dims4()?;
    let lk = k.dim(1)?;

    // Broadcast the single key head across all query heads, then batched matmul.
    let kt = k
        .transpose(D::Minus1, D::Minus2)? // [B, dh, Lk]
        .unsqueeze(1)? // [B, 1, dh, Lk]
        .broadcast_as((b, nh, dh, lk))?
        .contiguous()?; // [B, nh, dh, Lk]

    // q · k then ReLU clamp on the dot (never on the signed weight below).
    let dots = q.contiguous()?.matmul(&kt)?.relu()?; // [B, nh, Lq, Lk]

    let w = weights
        .transpose(1, 2)? // [B, nh, Lq]
        .unsqueeze(D::Minus1)? // [B, nh, Lq, 1]
        .contiguous()?;

    // Weight per head, then sum over heads -> [B, Lq, Lk].
    dots.broadcast_mul(&w)?.sum(1)
}

/// Select the top-`topk` keys per query row from a `[B, Lq, Lk]` score matrix.
///
/// When `causal`, query row `i` may only select keys `c <= i + (Lk - Lq)` (the
/// queries align with the last `Lq` positions of the `Lk`-length kv sequence);
/// each row then yields exactly `min(topk, num_valid_i)` selections.
fn select_topk(scores: &Tensor, topk: usize, causal: bool) -> Result<DsaSelection> {
    let (b, lq, lk) = scores.dims3()?;
    let dev = scores.device();
    let k_eff = topk.min(lk);

    let allowed = causal_allowed(lq, lk, dev)?; // [Lq, Lk] in {0, 1}

    let scores = if causal {
        // 0 where allowed, MASK_NEG_BIAS where not: allowed·(-NEG) + NEG.
        let bias = allowed.affine(-MASK_NEG_BIAS, MASK_NEG_BIAS)?;
        scores.broadcast_add(&bias.unsqueeze(0)?)?
    } else {
        scores.clone()
    };

    let indices = scores.contiguous()?.topk(k_eff)?.indices; // [B, Lq, k_eff] u32

    // Scatter ones at the selected indices to form the boolean mask.
    let ones = indices.ones_like()?.to_dtype(DType::F32)?;
    let mut mask =
        Tensor::zeros((b, lq, lk), DType::F32, dev)?.scatter_add(&indices, &ones, D::Minus1)?;

    // Drop any selection that landed on a causally-invalid position. This is
    // what makes rows with fewer valid candidates than `topk` keep exactly
    // `num_valid_i` selections (the masked-out top-k picks are zeroed here).
    if causal {
        mask = mask.broadcast_mul(&allowed.unsqueeze(0)?)?;
    }

    Ok(DsaSelection { indices, mask })
}

/// Causal "allowed" matrix `[Lq, Lk]` (`1.0` allowed, `0.0` masked). Queries
/// align with the last `Lq` positions of the `Lk`-length kv sequence.
fn causal_allowed(lq: usize, lk: usize, dev: &Device) -> Result<Tensor> {
    let offset = lk.saturating_sub(lq);
    let mut data = vec![0f32; lq * lk];
    for i in 0..lq {
        let last = (i + offset).min(lk - 1);
        for c in 0..=last {
            data[i * lk + c] = 1.0;
        }
    }
    Tensor::from_vec(data, (lq, lk), dev)
}

/// Apply partial RoPE to the indexer query (`[B, nh, Lq, dh]`) and key
/// (`[B, Lk, dh]`): the leading `rope_dim` dims of each head are rotated, the
/// rest pass through. Reuses the MLA rotary embedding (its cache is sized for
/// `qk_rope_head_dim == rope_dim`).
fn apply_partial_rope(
    q: &Tensor,
    k: &Tensor,
    rope_dim: usize,
    rotary: &DeepSeekV2RotaryEmbedding,
    positions: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let dh = q.dim(D::Minus1)?;
    let k4 = k.unsqueeze(1)?; // [B, 1, Lk, dh]

    if rope_dim >= dh {
        let (q, k4) = rotary.forward_positions(&q.contiguous()?, &k4.contiguous()?, positions)?;
        return Ok((q.contiguous()?, k4.squeeze(1)?.contiguous()?));
    }

    let q_rot = q.narrow(D::Minus1, 0, rope_dim)?.contiguous()?;
    let q_pass = q.narrow(D::Minus1, rope_dim, dh - rope_dim)?.contiguous()?;
    let k_rot = k4.narrow(D::Minus1, 0, rope_dim)?.contiguous()?;
    let k_pass = k4
        .narrow(D::Minus1, rope_dim, dh - rope_dim)?
        .contiguous()?;

    let (q_rot, k_rot) = rotary.forward_positions(&q_rot, &k_rot, positions)?;

    let q = Tensor::cat(&[&q_rot, &q_pass], D::Minus1)?.contiguous()?;
    let k4 = Tensor::cat(&[&k_rot, &k_pass], D::Minus1)?;
    Ok((q, k4.squeeze(1)?.contiguous()?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_quant::{ShardedSafeTensors, ShardedVarBuilder};
    use std::collections::HashMap;

    fn vb_from(tensors: HashMap<String, Tensor>) -> ShardedVarBuilder {
        ShardedSafeTensors::wrap(Box::new(tensors), DType::F32, Device::Cpu)
    }

    fn t2(data: Vec<f32>, rows: usize, cols: usize) -> Tensor {
        Tensor::from_vec(data, (rows, cols), &Device::Cpu).unwrap()
    }

    /// Toy indexer: `n_head=2, head_dim=2, topk=2`, no RoPE, no k_norm,
    /// `q_src_dim = kv_src_dim = 2`. The projections are chosen so the indexer's
    /// per-token outputs are hand-traceable:
    ///   * `wq_b` duplicates the input into both heads -> `q[i,h] = x[i]`.
    ///   * `wk` is identity -> `k[c] = x[c]`.
    ///   * `weights_proj` is identity -> `w[i] = x[i]` (then ·scale).
    fn toy_indexer() -> DsaIndexer {
        toy_indexer_topk(2)
    }

    fn toy_indexer_topk(index_topk: usize) -> DsaIndexer {
        let mut t = HashMap::new();
        // wq_b: [out=4, in=2] = [[1,0],[0,1],[1,0],[0,1]]
        t.insert(
            "wq_b.weight".to_string(),
            t2(vec![1., 0., 0., 1., 1., 0., 0., 1.], 4, 2),
        );
        // wk: identity [2,2]
        t.insert("wk.weight".to_string(), t2(vec![1., 0., 0., 1.], 2, 2));
        // weights_proj: identity [2,2]
        t.insert(
            "weights_proj.weight".to_string(),
            t2(vec![1., 0., 0., 1.], 2, 2),
        );

        let cfg = DsaConfig {
            index_n_heads: 2,
            index_head_dim: 2,
            index_topk,
            rope_dim: 0,
        };
        DsaIndexer::load(cfg, 2, 2, 1e-6, &None, vb_from(t))
            .unwrap()
            .expect("toy indexer tensors present")
    }

    /// Toy indexer with a key LayerNorm (`n_head=1, head_dim=2`), `wk` = identity
    /// so `keys(x)` exercises only the norm. `k_norm.weight = [1, 1]` and the
    /// given `bias`.
    fn toy_indexer_knorm(bias: [f32; 2]) -> DsaIndexer {
        let mut t = HashMap::new();
        t.insert("wq_b.weight".to_string(), t2(vec![1., 0., 0., 1.], 2, 2));
        t.insert("wk.weight".to_string(), t2(vec![1., 0., 0., 1.], 2, 2));
        t.insert("weights_proj.weight".to_string(), t2(vec![1., 0.], 1, 2));
        t.insert(
            "k_norm.weight".to_string(),
            Tensor::from_vec(vec![1f32, 1.], 2, &Device::Cpu).unwrap(),
        );
        t.insert(
            "k_norm.bias".to_string(),
            Tensor::from_vec(bias.to_vec(), 2, &Device::Cpu).unwrap(),
        );
        let cfg = DsaConfig {
            index_n_heads: 1,
            index_head_dim: 2,
            index_topk: 2,
            rope_dim: 0,
        };
        DsaIndexer::load(cfg, 2, 2, 1e-5, &None, vb_from(t))
            .unwrap()
            .expect("toy indexer tensors present")
    }

    /// The GGUF construction path: an indexer assembled via [`DsaIndexer::from_parts`]
    /// from freshly-built `QuantMethod` projections (as [`super::quantized_deepseek2`]
    /// does from GGUF `GgufMatMul` tensors) yields the **identical** selection to the
    /// var-builder [`DsaIndexer::load`] path — proving the decomplected constructor
    /// carries the same folded `weight_scale` and forward math, source-independently.
    #[test]
    fn from_parts_matches_load() {
        use hanzo_nn::Linear;
        use hanzo_quant::{QuantMethodConfig, UnquantLinear};

        let mk = |data: Vec<f32>, o: usize, i: usize| -> Arc<dyn QuantMethod> {
            let w = Tensor::from_vec(data, (o, i), &Device::Cpu).unwrap();
            Arc::new(
                UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(w, None))).unwrap(),
            )
        };
        let cfg = DsaConfig {
            index_n_heads: 2,
            index_head_dim: 2,
            index_topk: 2,
            rope_dim: 0,
        };
        // Same toy projections as `toy_indexer`, but built as QuantMethods.
        let idx = DsaIndexer::from_parts(
            cfg,
            mk(vec![1., 0., 0., 1., 1., 0., 0., 1.], 4, 2),
            mk(vec![1., 0., 0., 1.], 2, 2),
            mk(vec![1., 0., 0., 1.], 2, 2),
            None,
        );

        let x = Tensor::from_vec(vec![2f32, 0., 0., 3., 1., 2.], (1, 3, 2), &Device::Cpu).unwrap();
        let via_parts = idx.forward(&x, &x, None, None, true).unwrap();
        let via_load = toy_indexer().forward(&x, &x, None, None, true).unwrap();

        assert_eq!(
            via_parts.mask.to_vec3::<f32>().unwrap(),
            via_load.mask.to_vec3::<f32>().unwrap(),
            "from_parts selection mask must equal the load path"
        );
        assert_eq!(
            via_parts.indices.to_vec3::<u32>().unwrap(),
            via_load.indices.to_vec3::<u32>().unwrap(),
        );

        // Full selection (topk >= L): from_parts must still match load. Folding the
        // full causal selection into a causal base leaves it unchanged (the dense
        // token-for-token invariant), which is gold-tested via the load path in
        // `dsa_selection_drives_sdpa_and_full_selection_is_dense`.
        let full_parts = DsaIndexer::from_parts(
            DsaConfig {
                index_topk: 3,
                ..cfg
            },
            mk(vec![1., 0., 0., 1., 1., 0., 0., 1.], 4, 2),
            mk(vec![1., 0., 0., 1.], 2, 2),
            mk(vec![1., 0., 0., 1.], 2, 2),
            None,
        );
        assert_eq!(
            full_parts
                .forward(&x, &x, None, None, true)
                .unwrap()
                .mask
                .to_vec3::<f32>()
                .unwrap(),
            toy_indexer_topk(3)
                .forward(&x, &x, None, None, true)
                .unwrap()
                .mask
                .to_vec3::<f32>()
                .unwrap(),
        );
    }

    /// End-to-end proof of the ReLU·weight·sum + top-k math against fully
    /// hand-computed scores and selections, for both causal and non-causal.
    ///
    /// Input X = [[2,0],[0,3],[1,2]]. With the toy projections and
    /// `scale = 1/sqrt(2·2) = 0.5`, the score matrix works out to:
    ///   row0 = [4, 0, 2], row1 = [0, 13.5, 9], row2 = [3, 9, 7.5].
    #[test]
    fn forward_selection_matches_hand_computed() {
        let indexer = toy_indexer();
        let x = Tensor::from_vec(vec![2f32, 0., 0., 3., 1., 2.], (1, 3, 2), &Device::Cpu).unwrap();

        // --- Non-causal: every row sees all 3 keys, selects min(topk, L) = 2.
        let sel = indexer.forward(&x, &x, None, None, false).unwrap();
        assert_eq!(sel.mask.dims(), &[1, 3, 3], "mask shape [B, Lq, Lk]");
        assert_eq!(sel.indices.dims(), &[1, 3, 2], "indices shape [B, Lq, k]");

        assert_eq!(
            sel.mask.to_vec3::<f32>().unwrap(),
            vec![vec![
                vec![1., 0., 1.], // row0: keys {0,2} (scores 4,2 > 0)
                vec![0., 1., 1.], // row1: keys {1,2} (scores 13.5,9 > 0)
                vec![0., 1., 1.], // row2: keys {1,2} (scores 9,7.5 > 3)
            ]]
        );
        // Indices are deterministic here (no ties), sorted by descending score.
        assert_eq!(
            sel.indices.to_vec3::<u32>().unwrap(),
            vec![vec![vec![0, 2], vec![1, 2], vec![1, 2]]]
        );
        // Exactly min(topk, L) = 2 selected per row.
        for row in sel.mask.to_vec3::<f32>().unwrap()[0].iter() {
            assert_eq!(row.iter().sum::<f32>(), 2.0);
        }

        // --- Causal: row i selects min(topk, i+1) of keys 0..=i.
        let sel = indexer.forward(&x, &x, None, None, true).unwrap();
        assert_eq!(
            sel.mask.to_vec3::<f32>().unwrap(),
            vec![vec![
                vec![1., 0., 0.], // row0: only key 0 valid  -> count 1 = min(2,1)
                vec![1., 1., 0.], // row1: keys {0,1}         -> count 2 = min(2,2)
                vec![0., 1., 1.], // row2: top2 of {0,1,2}    -> count 2 = min(2,3)
            ]]
        );
        let counts: Vec<f32> = sel.mask.to_vec3::<f32>().unwrap()[0]
            .iter()
            .map(|r| r.iter().sum())
            .collect();
        assert_eq!(counts, vec![1.0, 2.0, 2.0]);
    }

    /// `indexer_scores` (the tensor path) must equal an independent scalar
    /// reference implementing `Σ_h ReLU(dot)·w` — proving the ReLU clamp,
    /// per-head weighting and head-sum, including signed weights and negative
    /// dots that the ReLU must clamp to zero.
    #[test]
    fn scores_match_scalar_reference() {
        let dev = Device::Cpu;
        let (nh, lq, dh, lk) = (2usize, 2usize, 2usize, 3usize);

        // Deterministic values with mixed signs so some dots are negative.
        let q_data = vec![
            // head 0: q[0]=[1,-2], q[1]=[0.5,1]
            1.0f32, -2.0, 0.5, 1.0, // head 1: q[0]=[2,0], q[1]=[-1,3]
            2.0, 0.0, -1.0, 3.0,
        ];
        let k_data = vec![1.0f32, 1.0, -2.0, 0.5, 0.0, -3.0]; // k[0..3], dh=2
        let w_data = vec![0.5f32, -1.0, 2.0, 0.25]; // [Lq, nh]

        let q = Tensor::from_vec(q_data.clone(), (1, nh, lq, dh), &dev).unwrap();
        let k = Tensor::from_vec(k_data.clone(), (1, lk, dh), &dev).unwrap();
        let w = Tensor::from_vec(w_data.clone(), (1, lq, nh), &dev).unwrap();

        let scores = indexer_scores(&q, &k, &w)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();

        for i in 0..lq {
            for c in 0..lk {
                let mut s = 0.0f32;
                for h in 0..nh {
                    let mut dot = 0.0f32;
                    for d in 0..dh {
                        dot += q_data[h * lq * dh + i * dh + d] * k_data[c * dh + d];
                    }
                    s += dot.max(0.0) * w_data[i * nh + h]; // ReLU on dot, signed w
                }
                assert!(
                    (scores[0][i][c] - s).abs() < 1e-5,
                    "score[{i},{c}] tensor={} scalar={s}",
                    scores[0][i][c]
                );
            }
        }
    }

    /// The ReLU clamp is applied to the dot product (a negative dot contributes
    /// `0`, not its signed value).
    #[test]
    fn relu_clamps_negative_dot() {
        let dev = Device::Cpu;
        // 1 head, dh=1, 1 query, 2 keys. q=1; k0=2, k1=-3; w=1.
        let q = Tensor::from_vec(vec![1.0f32], (1, 1, 1, 1), &dev).unwrap();
        let k = Tensor::from_vec(vec![2.0f32, -3.0], (1, 2, 1), &dev).unwrap();
        let w = Tensor::from_vec(vec![1.0f32], (1, 1, 1), &dev).unwrap();

        let scores = indexer_scores(&q, &k, &w)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        // dot(k0,q)=2 -> 2 ; dot(k1,q)=-3 -> ReLU -> 0 (NOT -3).
        assert_eq!(scores[0][0], vec![2.0, 0.0]);
    }

    /// `topk >= kv_len` selects everything (clamped to `kv_len`), and the
    /// additive bias is `0` for selected / large-negative for masked keys.
    #[test]
    fn topk_clamps_and_bias_is_correct() {
        let dev = Device::Cpu;
        // scores [B=1, Lq=2, Lk=2]; ask for topk=5 (> Lk).
        let scores = Tensor::from_vec(vec![3.0f32, 1.0, 0.0, 4.0], (1, 2, 2), &dev).unwrap();
        let sel = select_topk(&scores, 5, false).unwrap();
        assert_eq!(
            sel.mask.to_vec3::<f32>().unwrap(),
            vec![vec![vec![1., 1.], vec![1., 1.]]],
            "topk clamped to kv_len selects all"
        );

        let bias = sel
            .to_attn_bias(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        for row in &bias[0] {
            for v in row {
                assert_eq!(*v, 0.0, "all selected -> zero bias");
            }
        }

        // A partial selection yields a large-negative bias on the dropped key.
        let sel = select_topk(&scores, 1, false).unwrap();
        let bias = sel
            .to_attn_bias(DType::F32)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert_eq!(bias[0][0][0], 0.0); // key0 selected in row0 (score 3 > 1)
        assert!(bias[0][0][1] < -1e29); // key1 dropped -> -inf-ish
    }

    /// FIX 1 regression guard: the indexer key norm is a real LayerNorm
    /// (mean-centered + learned bias), NOT an RmsNorm. Under RmsNorm the bias
    /// would be silently dropped and there would be no mean-centering — both of
    /// which this test would catch.
    #[test]
    fn k_norm_is_layernorm_not_rmsnorm() {
        let dev = Device::Cpu;
        // Single key x = [3, 1] (mean = 2, so mean-centering is observable).
        let x = Tensor::from_vec(vec![3f32, 1.], (1, 1, 2), &dev).unwrap();

        let keys0 = toy_indexer_knorm([0.0, 0.0])
            .keys(&x)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        // LayerNorm: (x - mean)/std -> ~[1, -1]. The NEGATIVE entry is impossible
        // under RmsNorm (positive input / positive RMS stays positive).
        assert!((keys0[0][0][0] - 1.0).abs() < 1e-3, "got {:?}", keys0[0][0]);
        assert!((keys0[0][0][1] + 1.0).abs() < 1e-3, "got {:?}", keys0[0][0]);
        // RmsNorm(x)[1] would be 1/sqrt(mean(x^2)) = 1/sqrt(5) ~ +0.447.
        let rms_entry1 = 1.0f32 / 5.0f32.sqrt();
        assert!(
            (keys0[0][0][1] - rms_entry1).abs() > 0.5,
            "LayerNorm keys must differ from RmsNorm"
        );

        // A non-uniform, non-zero bias shifts every key by exactly that bias
        // (RmsNorm would drop the bias -> zero difference).
        let keys_b = toy_indexer_knorm([0.5, -0.5])
            .keys(&x)
            .unwrap()
            .to_vec3::<f32>()
            .unwrap();
        assert!((keys_b[0][0][0] - (keys0[0][0][0] + 0.5)).abs() < 1e-5);
        assert!((keys_b[0][0][1] - (keys0[0][0][1] - 0.5)).abs() < 1e-5);
    }

    /// FIX 3 end-to-end: a DSA selection — folded into the causal mask via the
    /// real [`DsaSelection::combine_with_mask`] and run through the real
    /// [`crate::attention::Sdpa`] — actually changes the attention output; and a
    /// full selection (`topk >= L`, zero bias) leaves it identical to dense.
    ///
    /// `ModelForwardContext` is not unit-constructible anywhere in the engine, so
    /// this exercises the exact cold-prefill wiring at the SDPA boundary rather
    /// than via `Attention::forward`.
    #[test]
    fn dsa_selection_drives_sdpa_and_full_selection_is_dense() {
        use crate::attention::{AttentionMask, Sdpa, SdpaParams};
        let dev = Device::Cpu;
        let (b, h, l, hd) = (1usize, 1usize, 3usize, 4usize);

        let mk = |seed: f32| {
            let data: Vec<f32> = (0..b * h * l * hd)
                .map(|i| ((i as f32) * 0.37 + seed).sin())
                .collect();
            Tensor::from_vec(data, (b, h, l, hd), &dev).unwrap()
        };
        let (q, k, v) = (mk(0.0), mk(1.0), mk(2.0));
        let params = SdpaParams {
            n_kv_groups: 1,
            softcap: None,
            softmax_scale: 1.0 / (hd as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        };

        // Additive causal base mask [L, L]: 0 on/below diagonal, -inf above.
        let mut base_data = vec![0f32; l * l];
        for i in 0..l {
            for c in (i + 1)..l {
                base_data[i * l + c] = f32::NEG_INFINITY;
            }
        }
        let base = Tensor::from_vec(base_data, (l, l), &dev).unwrap();

        // Indexer hidden, L=3 > index_topk=2 -> causal row 2 drops one of 3 keys.
        let x = Tensor::from_vec(vec![2f32, 0., 0., 3., 1., 2.], (1, l, 2), &dev).unwrap();

        let sel = toy_indexer().forward(&x, &x, None, None, true).unwrap();
        assert!(
            (sel.mask.to_vec3::<f32>().unwrap()[0][2].iter().sum::<f32>()) < l as f32,
            "DSA must mask at least one causally-valid key on the last row"
        );

        let dense = Sdpa
            .run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::Custom(base.clone()),
                None,
                &params,
            )
            .unwrap();
        let dsa = Sdpa
            .run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::Custom(sel.combine_with_mask(&base).unwrap()),
                None,
                &params,
            )
            .unwrap();

        // (a) DSA selection changes the attention output through the real Sdpa.
        let dv = dense.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let sv = dsa.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let max_diff = dv
            .iter()
            .zip(&sv)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-4,
            "DSA selection must change the output (max_diff={max_diff})"
        );

        // (b) Full selection (topk >= L) -> zero bias -> identical to dense.
        let sel_full = toy_indexer_topk(l)
            .forward(&x, &x, None, None, true)
            .unwrap();
        let dsa_full = Sdpa
            .run_attention(
                &q,
                &k,
                &v,
                &AttentionMask::Custom(sel_full.combine_with_mask(&base).unwrap()),
                None,
                &params,
            )
            .unwrap();
        let fv = dsa_full.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (a, c) in dv.iter().zip(&fv) {
            assert!(
                (a - c).abs() < 1e-6,
                "full selection must equal dense: {a} vs {c}"
            );
        }
    }
}
