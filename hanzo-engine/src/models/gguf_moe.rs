#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Shared GGUF fused-MoE block for the DeepSeek-V2/V3 and GLM-4.5/4.7 families.
//!
//! Both `deepseek2` and `glm4moe` GGUFs stack the routed experts identically
//! (`ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_{gate,up,down}_shexp`, `exp_probs_b.bias`)
//! and select them with the same sigmoid/softmax + optional group-limited no-aux-loss gate. This is
//! the one place that math lives; the arch models supply only their per-family params + attention.

use std::sync::Arc;

use crate::gguf::Content;
use crate::ops::{TopKLastDimOp, TopKOutput};
use hanzo_ml::quantized::QMatMul;
use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

pub(crate) fn gguf_linear(q: hanzo_ml::quantized::QTensor) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(q),
        b: None,
    })?))
}

pub(crate) struct Mlp {
    gate: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    down: Arc<dyn QuantMethod>,
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate.forward(xs)?;
        let up = self.up.forward(xs)?;
        let y = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        self.down.forward(&y)
    }
}

pub(crate) struct MoeGate {
    weight: Tensor,
    e_score_correction_bias: Option<Tensor>,
    top_k: usize,
    n_routed_experts: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f64,
    norm_topk_prob: bool,
    sigmoid_scoring: bool,
}

impl MoeGate {
    // (topk_idx, topk_weight). Greedy softmax (V2) or grouped sigmoid no-aux (V3/GLM) selection.
    fn forward(&self, xs: &Tensor) -> Result<(Tensor, Tensor)> {
        let (bs, seq_len, h) = xs.dims3()?;
        let xs = xs.reshape(((), h))?;
        // Router weight is held f16 (`build_moe_or_mlp` below): half the resident bytes and half the
        // per-token read of the [n_experts, hidden] matrix. The matmul reads f16 and accumulates the
        // small [tokens, n_experts] logits back to f32 so scoring/top-k stay f32-stable. f16 rounding
        // (~2^-11 relative) sits well under the Q4/Q6 quantization noise the gate already carries.
        let logits = xs
            .to_dtype(self.weight.dtype())?
            .broadcast_matmul(&self.weight.t()?)?
            .to_dtype(DType::F32)?;
        let scores = if self.sigmoid_scoring {
            // 1/(1+exp(-x)) via standard unaries; the fused Sigmoid custom-op has no ROCm kernel.
            (logits.neg()?.exp()? + 1.0)?.recip()?
        } else {
            hanzo_nn::ops::softmax_last_dim(&logits)?
        };

        let mut topk_weight;
        let topk_idx;
        if let Some(bias) = &self.e_score_correction_bias {
            let scores_for_choice = scores
                .reshape((bs * seq_len, ()))?
                .broadcast_add(&bias.unsqueeze(0)?)?;
            // With a single expert group the group-limited mask is all-ones (identity), so skip the
            // scatter entirely (also sidesteps ROCm's missing scatter_add). n_group > 1 keeps the
            // DeepSeek-V3 grouped no-aux-loss selection.
            let tmp_scores = if self.n_group <= 1 {
                scores_for_choice
            } else {
                let group_scores = scores_for_choice
                    .reshape((bs * seq_len, self.n_group, ()))?
                    .topk(2)?
                    .values
                    .sum(D::Minus1)?;
                let group_idx = group_scores.topk(self.topk_group)?.indices;
                let mut group_mask = group_scores.zeros_like()?;
                group_mask = group_mask.scatter_add(
                    &group_idx,
                    &group_idx.ones_like()?.to_dtype(group_mask.dtype())?,
                    1,
                )?;
                let score_mask = group_mask
                    .unsqueeze(D::Minus1)?
                    .expand((
                        bs * seq_len,
                        self.n_group,
                        self.n_routed_experts / self.n_group,
                    ))?
                    .reshape((bs * seq_len, ()))?;
                scores_for_choice.broadcast_mul(&score_mask)?
            };
            topk_idx = tmp_scores.topk(self.top_k)?.indices;
            topk_weight = scores.gather(&topk_idx, 1)?;
        } else {
            let TopKOutput { values, indices } = scores.topk(self.top_k)?;
            topk_weight = values;
            topk_idx = indices;
        }

        if self.norm_topk_prob {
            let denom = (topk_weight.sum_keepdim(D::Minus1)? + 1e-20)?;
            topk_weight = topk_weight.broadcast_div(&denom)?;
        }
        topk_weight = (topk_weight * self.routed_scaling_factor)?;
        Ok((topk_idx, topk_weight))
    }
}

pub(crate) struct FusedMoe {
    gate: MoeGate,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    shared: Option<Mlp>,
}

#[cfg(feature = "cuda")]
fn moe_qtensor(q: &QMatMul) -> Option<&hanzo_ml::quantized::QTensor> {
    match q {
        QMatMul::QTensor(qt) => Some(qt),
        _ => None,
    }
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let identity = xs.clone();
        let xs_flat = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs_flat.dtype();
        let num_tokens = xs_flat.dim(0)?;

        let (topk_idx, topk_weight) = self.gate.forward(xs)?;

        let mut y = self
            .experts(
                &xs_flat,
                &topk_idx,
                &topk_weight,
                num_tokens,
                original_dtype,
            )?
            .reshape((batch, seq_len, hidden_dim))?;

        if let Some(shared) = &self.shared {
            y = (y + shared.forward(&identity)?)?;
        }
        Ok(y)
    }

    // Routed expert compute -> [num_tokens, hidden] in original_dtype. Prefill routes the 16-bit
    // activation straight through the fused q8_1 MMQ grouped-GEMM (no f32 round-trip); decode and
    // short prompts keep the per-slot indexed matvec (capture-clean, no host-side counting sort).
    fn experts(
        &self,
        xs_flat: &Tensor,
        topk_idx: &Tensor,
        topk_weight: &Tensor,
        num_tokens: usize,
        original_dtype: DType,
    ) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if num_tokens >= crate::moe::grouped::GROUPED_MIN_TOKENS {
            if let (Some(g), Some(u), Some(d)) = (
                moe_qtensor(&self.gate_experts),
                moe_qtensor(&self.up_experts),
                moe_qtensor(&self.down_experts),
            ) {
                if let Some(y) = crate::moe::grouped::moe_grouped_prefill(
                    g,
                    u,
                    d,
                    xs_flat,
                    topk_idx,
                    topk_weight,
                    crate::layers::Activation::Silu,
                    self.gate.n_routed_experts,
                    self.gate.top_k,
                    original_dtype,
                )? {
                    return Ok(y);
                }
            }
        }

        let hidden_dim = xs_flat.dim(1)?;
        // Decode stays single-dtype end-to-end: the routed activation rides the model's native
        // bf16/f16 through gate+up+down (the down matvec and both fused banks consume it directly),
        // dropping the F32 round-trip that used to wrap every gate/up/down matvec. `moe_combine`
        // reduces the per-expert outputs by the router weights in one pass (f32 accumulate) rather
        // than a bf16 broadcast-mul + strided sum.
        //
        // Sharing ONE broadcast+quantize of the routed token across gate and up is a ROCm-only
        // property: that fusion is `#[cfg(feature = "rocm")]` in `moe_gate_up` (quantized/mod.rs).
        // Every other backend falls through to two independent `indexed_moe_forward` calls, and the
        // Vulkan dp4a path re-quantizes the identical activation inside each one.
        let xs3 = xs_flat.reshape((num_tokens, 1, hidden_dim))?;
        // Unquantized (e.g. all-F32) GGUF: the expert bank dequantizes to a plain `Tensor`, which the
        // fused `moe_gate_up`/`indexed_moe_forward` quant kernels don't cover. Run the dense per-expert
        // twin instead (same gather->matmul->scatter contract), keeping the quantized path untouched.
        if let (Some(g), Some(u), Some(d)) = (
            dense_bank(&self.gate_experts),
            dense_bank(&self.up_experts),
            dense_bank(&self.down_experts),
        ) {
            let gate = dense_indexed_moe(&g, &xs3, topk_idx)?;
            let up = dense_indexed_moe(&u, &xs3, topk_idx)?;
            let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
            let ys = dense_indexed_moe(&d, &activated, topk_idx)?;
            return hanzo_ml::quantized::moe_combine(&ys, topk_weight)?.to_dtype(original_dtype);
        }
        let (gate, up) =
            hanzo_ml::quantized::moe_gate_up(&xs3, topk_idx, &self.gate_experts, &self.up_experts)?;
        let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
        let ys = self
            .down_experts
            .indexed_moe_forward(&activated, topk_idx)?;
        hanzo_ml::quantized::moe_combine(&ys, topk_weight)?.to_dtype(original_dtype)
    }
}

// Dequantized expert bank behind a `QMatMul`, if the weight is stored dense (`Tensor`/`TensorF16`).
fn dense_bank(q: &QMatMul) -> Option<Tensor> {
    match q {
        QMatMul::Tensor(t) | QMatMul::TensorF16(t) => Some(t.clone()),
        _ => None,
    }
}

// Dense twin of `QTensor::indexed_moe_forward` for an unquantized `[E, n, k]` expert bank.
// `x` is `[t, s, k]` with `s == 1` (shared input, broadcast over topk) or `s == topk` (per-slot);
// `ids` is `[t, topk]`. Returns `[t, topk, n]`. Groups routed slots by expert, one matmul each.
fn dense_indexed_moe(bank: &Tensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    use std::collections::HashMap;
    let device = x.device();
    let out_dtype = x.dtype();
    let (_e_cnt, n, k) = bank.dims3()?;
    let (t, topk) = ids.dims2()?;
    let s = x.dim(1)?;
    let x_exp = if s == topk {
        x.clone()
    } else {
        x.broadcast_as((t, topk, k))?
    };
    let x_flat = x_exp
        .reshape((t * topk, k))?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let bank = bank.to_dtype(DType::F32)?;
    let ids_vec = ids
        .reshape((t * topk,))?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;
    let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
    for (slot, eid) in ids_vec.iter().enumerate() {
        groups.entry(*eid).or_default().push(slot as u32);
    }
    let mut out_flat = Tensor::zeros((t * topk, n), DType::F32, device)?;
    for (eid, slots) in groups {
        let w_e = bank.narrow(0, eid as usize, 1)?.squeeze(0)?; // [n, k]
        let idx = Tensor::from_vec(slots.clone(), (slots.len(),), device)?;
        let x_e = x_flat.index_select(&idx, 0)?; // [m, k]
        let y_e = x_e.matmul(&w_e.t()?)?; // [m, n]
        out_flat = out_flat.index_add(&idx, &y_e, 0)?;
    }
    out_flat.reshape((t, topk, n))?.to_dtype(out_dtype)
}

pub(crate) enum MoeOrMlp {
    FusedMoe(Box<FusedMoe>),
    Mlp(Mlp),
}

impl MoeOrMlp {
    pub(crate) fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

/// Per-family routing/expert config, all sourced from GGUF metadata.
pub(crate) struct MoeParams {
    pub n_routed_experts: usize,
    pub num_experts_per_tok: usize,
    pub n_group: usize,
    pub topk_group: usize,
    pub routed_scaling_factor: f64,
    pub norm_topk_prob: bool,
    pub sigmoid_scoring: bool,
    pub n_shared_experts: usize,
    pub leading_dense_block_count: usize,
}

/// Build the layer's feed-forward block: a dense SwiGLU MLP for the leading dense layers, else the
/// fused routed+shared MoE. `blk.{layer_idx}` tensors are read from `ct`.
pub(crate) fn build_moe_or_mlp<R: std::io::Seek + std::io::Read>(
    ct: &mut Content<'_, R>,
    layer_idx: usize,
    device: &Device,
    p: &MoeParams,
) -> Result<MoeOrMlp> {
    let prefix = format!("blk.{layer_idx}");
    let is_moe = p.n_routed_experts > 0 && layer_idx >= p.leading_dense_block_count;
    if !is_moe {
        return Ok(MoeOrMlp::Mlp(Mlp {
            gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?)?,
            up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?)?,
            down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?)?,
        }));
    }

    let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
    let gate_experts = ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?;
    let up_experts = ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;
    let down_experts = ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
    let e_score_correction_bias =
        if p.sigmoid_scoring && ct.has_tensor(&format!("{prefix}.exp_probs_b.bias")) {
            Some(
                ct.tensor(&format!("{prefix}.exp_probs_b.bias"), device)?
                    .dequantize(device)?
                    .to_dtype(DType::F32)?,
            )
        } else {
            None
        };
    let shared = if p.n_shared_experts > 0 {
        Some(Mlp {
            gate: gguf_linear(ct.tensor(&format!("{prefix}.ffn_gate_shexp.weight"), device)?)?,
            up: gguf_linear(ct.tensor(&format!("{prefix}.ffn_up_shexp.weight"), device)?)?,
            down: gguf_linear(ct.tensor(&format!("{prefix}.ffn_down_shexp.weight"), device)?)?,
        })
    } else {
        None
    };
    Ok(MoeOrMlp::FusedMoe(Box::new(FusedMoe {
        gate: MoeGate {
            weight: gate.dequantize_f16(device)?,
            e_score_correction_bias,
            top_k: p.num_experts_per_tok,
            n_routed_experts: p.n_routed_experts,
            n_group: p.n_group,
            topk_group: p.topk_group,
            routed_scaling_factor: p.routed_scaling_factor,
            norm_topk_prob: p.norm_topk_prob,
            sigmoid_scoring: p.sigmoid_scoring,
        },
        gate_experts: QMatMul::from_qtensor(gate_experts)?,
        up_experts: QMatMul::from_qtensor(up_experts)?,
        down_experts: QMatMul::from_qtensor(down_experts)?,
        shared,
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic N(0,1) sample stream. The CPU backend's Device::set_seed is a no-op (candle's CPU
    // rng is not seedable), so the guards below drive their own seeded rng and build tensors from the
    // values -- otherwise unseeded randn lets a borderline top-k boundary flip under f16 rounding and
    // the guard flakes for reasons unrelated to what it guards.
    fn normals(seed: u64, n: usize) -> Vec<f32> {
        use rand::prelude::*;
        use rand_distr::StandardNormal;
        let mut rng = rand_isaac::Isaac64Rng::seed_from_u64(seed);
        (0..n)
            .map(|_| rng.sample::<f32, _>(StandardNormal))
            .collect()
    }

    // Guards the gguf_moe.rs router-weight f16 hold: the [tokens, n_experts] logits still accumulate
    // f32-stable, so top-k expert selection is unchanged vs the old f32 weight at GLM/DeepSeek scale.
    #[test]
    fn f16_router_preserves_topk() -> Result<()> {
        let dev = Device::Cpu;
        let (tokens, hidden, experts, top_k) = (8usize, 5120usize, 256usize, 8usize);
        // Router logits ~ N(0, hidden * var); scale the weight so logits land in the usual O(1)
        // range instead of a degenerate tie band.
        let scale = 1.0 / (hidden as f64).sqrt();
        let w = (Tensor::from_vec(normals(0x9E37, experts * hidden), (experts, hidden), &dev)?
            * scale)?;
        let x = Tensor::from_vec(normals(0x1F83, tokens * hidden), (tokens, hidden), &dev)?;

        let logits_f32 = x.broadcast_matmul(&w.t()?)?;
        let w16 = w.to_dtype(DType::F16)?;
        let logits_f16 = x
            .to_dtype(DType::F16)?
            .broadcast_matmul(&w16.t()?)?
            .to_dtype(DType::F32)?;

        let top_f32 = logits_f32.topk(top_k)?.indices.to_vec2::<u32>()?;
        let top_f16 = logits_f16.topk(top_k)?.indices.to_vec2::<u32>()?;
        assert_eq!(top_f32, top_f16, "f16 router changed top-{top_k} routing");
        Ok(())
    }

    // Guards the decode expert-glue rewrite: the fused single-dtype path (moe_gate_up shared-quantize
    // + moe_combine reduce, activation kept in its native dtype) reproduces the prior two-matvec +
    // F32-cast + broadcast-mul/sum reference on a FIXED routed selection. Routing (topk_idx) is held
    // constant, so this isolates the expert-compute change from the top-k selection guarded above.
    // On CPU moe_gate_up runs its unfused fallback, so this also pins the "fallback == two separate
    // indexed_moe_forward" contract; the ROCm shared-quantize kernel is bit-identical by construction
    // (the q8_1 activation is deterministic in x).
    #[test]
    fn fused_decode_matches_unfused_reference() -> Result<()> {
        use hanzo_ml::quantized::{GgmlDType, QMatMul, QTensor};
        let dev = Device::Cpu;
        let (tokens, topk, experts, hidden, inter) = (3usize, 2usize, 4usize, 64usize, 32usize);
        let bank = |seed: u64, rows: usize, cols: usize| -> Result<QMatMul> {
            let w = (Tensor::from_vec(
                normals(seed, experts * rows * cols),
                (experts, rows, cols),
                &dev,
            )? * 0.5)?;
            QMatMul::from_qtensor(QTensor::quantize(&w, GgmlDType::Q4_0)?)
        };
        let gate = bank(0xA1, inter, hidden)?;
        let up = bank(0xB2, inter, hidden)?;
        let down = bank(0xC3, hidden, inter)?;

        let xs = Tensor::from_vec(normals(0xD4, tokens * hidden), (tokens, hidden), &dev)?;
        let idx = Tensor::from_vec(vec![0u32, 1, 2, 3, 1, 0], (tokens, topk), &dev)?;
        let weight = Tensor::from_vec(normals(0xE5, tokens * topk), (tokens, topk), &dev)?.abs()?;

        // New path: mirror of experts() decode -- native dtype, one fused gate+up, fused combine.
        let xs3 = xs.reshape((tokens, 1, hidden))?;
        let (g, u) = hanzo_ml::quantized::moe_gate_up(&xs3, &idx, &gate, &up)?;
        let act = crate::ops::mul_and_act(&g, &u, crate::layers::Activation::Silu)?;
        let ys = down.indexed_moe_forward(&act, &idx)?;
        let new = hanzo_ml::quantized::moe_combine(&ys, &weight)?;

        // Prior reference: F32 force-cast, two independent matvecs, broadcast-mul + strided sum.
        let xs3f = xs.to_dtype(DType::F32)?.reshape((tokens, 1, hidden))?;
        let g2 = gate.indexed_moe_forward(&xs3f, &idx)?;
        let u2 = up.indexed_moe_forward(&xs3f, &idx)?;
        let act2 = crate::ops::mul_and_act(&g2, &u2, crate::layers::Activation::Silu)?;
        let ys2 = down.indexed_moe_forward(&act2, &idx)?;
        let old = ys2
            .broadcast_mul(&weight.to_dtype(ys2.dtype())?.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?;

        let diff = (new - old)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            diff < 1e-5,
            "fused decode diverged from reference by {diff}"
        );
        Ok(())
    }
}
