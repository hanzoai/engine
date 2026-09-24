#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
// The first caller is the qwen4exp loader.
#![allow(dead_code)]

//! Qwen4Exp QSA: attention restricted to the best compressed key groups.
//!
//! A weight-free indexer projects `heads` query heads and one key head of width `dim` from the
//! block input. Each run of `ratio` raw keys pools into one compressed key. A query scores every
//! visible group by `Σ_h ReLU(q_h·kc_j)/√dim`, keeps the best `blocks`, and attends their tokens
//! plus its own open tail group: at most `width = blocks·ratio + ratio - 1` tokens, so QSA is
//! exactly dense attention for sequences of up to `width` tokens.
//!
//! Line references are to vLLM 0.29.0 `vllm/models/qwen4_exp/` and to the deterministic top-k
//! served with `VLLM_QSA_DET_TOPK=1` (`kernel-det/src/persistent_topk.cuh`). This is the CPU
//! implementation.

use std::io::{Read, Seek};
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_quant::QuantMethod;

use crate::gguf::Content;
use crate::layers::QRmsNorm;
use crate::models::quantized_qwen3_5_moe::gguf_qmm;
use crate::utils::gguf_metadata::ContentMetadata;

/// Attention with the QSA kernel's math over dense causal keys, which is what QSA computes while
/// every visible key is selected (`ops/qsa.py:193-359`, one tile): f32 scores scaled by
/// `d^-0.5 · log2 e`, `exp2` against the row max, the probabilities rounded to the activation
/// dtype for the PV product while the normalizer sums them unrounded, an f32 PV, one division and
/// one rounding.
///
/// `q` `[b, hq, s, d]`, `k`/`v` `[b, hkv, L, d]` with the `s` queries at positions `L - s ..
/// L`; returns `[b, hq, s, d]` in `q`'s dtype.
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
    let dtype = q.dtype();
    let (b, hq, s, d) = q.dims4()?;
    let (_, hkv, l, _) = k.dims4()?;
    let group = hq / hkv;
    let expand = |t: &Tensor| -> Result<Tensor> {
        if group == 1 {
            return t.to_dtype(DType::F32);
        }
        t.to_dtype(DType::F32)?
            .unsqueeze(2)?
            .broadcast_as((b, hkv, group, l, d))?
            .reshape((b, hq, l, d))
    };
    let (k, v) = (expand(k)?, expand(v)?);
    let scale = ((d as f64).powf(-0.5) * std::f64::consts::LOG2_E) as f32;
    let scores = (q.to_dtype(DType::F32)?.matmul(&k.t()?.contiguous()?)? * f64::from(scale))?;
    let offset = l - s;
    let mask: Vec<f32> = (0..s)
        .flat_map(|i| (0..l).map(move |j| if j <= offset + i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, s, l), q.device())?;
    let scores = scores.broadcast_add(&mask)?;
    let max = scores.max_keepdim(D::Minus1)?;
    let p = (scores.broadcast_sub(&max)? * std::f64::consts::LN_2)?.exp()?;
    let l_sum = p.sum_keepdim(D::Minus1)?;
    let p = p.to_dtype(dtype)?.to_dtype(DType::F32)?;
    p.matmul(&v.contiguous()?)?
        .broadcast_div(&l_sum)?
        .to_dtype(dtype)
}

/// Shape of one layer's indexer and of its selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    /// Query heads.
    heads: usize,
    /// Width of each query head and of the single key head.
    dim: usize,
    /// Tokens pooled into one compressed key.
    ratio: usize,
    /// Groups kept per query.
    blocks: usize,
}

impl Config {
    /// Layer `layer`'s indexer from `attention.indexer.{head_count, key_length, top_k}` and its
    /// entry of `attention.compress_ratios`; `None` when that entry is 0 or absent (a GDN layer, or
    /// the MTP layer past the table). `top_k` counts tokens, so `blocks = top_k / ratio`
    /// (nvidia/indexer_qsa.py:117-121).
    pub(crate) fn from_gguf(md: &ContentMetadata, layer: usize) -> Result<Option<Self>> {
        let msg = |e: anyhow::Error| hanzo_ml::Error::Msg(e.to_string());
        let ratio = match md
            .get_value::<Vec<i32>>("attention.compress_ratios")
            .map_err(msg)?
            .get(layer)
        {
            None | Some(0) => return Ok(None),
            Some(&r) => usize::try_from(r)
                .map_err(|_| hanzo_ml::Error::Msg(format!("layer {layer}: compress ratio {r}")))?,
        };
        let get = |key: &str| md.get_value::<u32>(key).map(|v| v as usize).map_err(msg);
        let (heads, dim, tokens) = (
            get("attention.indexer.head_count")?,
            get("attention.indexer.key_length")?,
            get("attention.indexer.top_k")?,
        );
        // config.py:143-151: every value positive and the budget a whole number of groups.
        if heads == 0 || dim == 0 || tokens == 0 || tokens % ratio != 0 {
            hanzo_ml::bail!(
                "layer {layer}: QSA heads {heads}, dim {dim}, top_k {tokens}, ratio {ratio}"
            );
        }
        Ok(Some(Self {
            heads,
            dim,
            ratio,
            blocks: tokens / ratio,
        }))
    }

    /// Selection width: `blocks` whole groups plus a tail of at most `ratio - 1` tokens
    /// (nvidia/indexer_qsa.py:170-172).
    pub(crate) fn width(&self) -> usize {
        self.blocks * self.ratio + self.ratio - 1
    }
}

/// One layer's indexer. The q/k norms are Gemma norms whose `1 + w` the GGUF already carries, so
/// they run as plain RMS norms, in f32 with that f32 weight.
pub(crate) struct Indexer {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    cfg: Config,
}

impl Indexer {
    /// `{prefix}.indexer.{q_proj, k_proj, q_norm, k_norm}`: the checkpoint's `index_qk_proj`
    /// split into its query rows and its key rows (nvidia/indexer_qsa.py:134-148, 230-237).
    pub(crate) fn from_gguf<R: Read + Seek>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        cfg: Config,
        eps: f32,
        dev: &Device,
    ) -> Result<Self> {
        let mut tensor = |name: &str| ct.tensor(&format!("{prefix}.indexer.{name}.weight"), dev);
        Ok(Self {
            q: gguf_qmm(tensor("q_proj")?)?,
            k: gguf_qmm(tensor("k_proj")?)?,
            q_norm: QRmsNorm::new(tensor("q_norm")?, eps)?,
            k_norm: QRmsNorm::new(tensor("k_norm")?, eps)?,
            cfg,
        })
    }

    /// Queries `[B, S, heads, dim]`, normed in f32 and rounded once, then roped, and raw keys
    /// `[B, S, dim]` of `u` `[B, S, hidden]`; `cos_sin` holds the rotary tables at the chunk's
    /// positions (nvidia/indexer_qsa.py:230-237, 276-282; nvidia/ops/qsa_pre_indexer.py:68-71).
    pub(crate) fn project(
        &self,
        u: &Tensor,
        cos_sin: &(Tensor, Tensor),
    ) -> Result<(Tensor, Tensor)> {
        let (b, s, _) = u.dims3()?;
        let q = self
            .q
            .forward(u)?
            .reshape((b, s, self.cfg.heads, self.cfg.dim))?
            .transpose(1, 2)?;
        let normed = self
            .q_norm
            .forward(&q.to_dtype(DType::F32)?)?
            .to_dtype(q.dtype())?;
        let q = rope(&normed, cos_sin)?.transpose(1, 2)?;
        Ok((q, self.k.forward(u)?))
    }

    /// Compressed keys `[G, dim]` of raw keys `[G·ratio, dim]` taken group by group: the f32
    /// mean, rounded to bf16, normed in f32 and rounded once to the raw keys' dtype, then roped at
    /// the group's first position `ratio·j`, where `cos_sin` holds one row per group
    /// (nvidia/ops/qsa_pre_indexer.py:68-71, 261-269, 271, 322-334).
    pub(crate) fn compress(&self, raw: &Tensor, cos_sin: &(Tensor, Tensor)) -> Result<Tensor> {
        let (n, dim) = raw.dims2()?;
        let r = self.cfg.ratio;
        if n % r != 0 {
            hanzo_ml::bail!("{n} raw keys do not fill groups of {r}");
        }
        let g = n / r;
        let pooled = (raw.to_dtype(DType::F32)?.reshape((g, r, dim))?.sum(1)? / r as f64)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        let x = self
            .k_norm
            .forward(&pooled)?
            .to_dtype(raw.dtype())?
            .reshape((1, 1, g, dim))?;
        rope(&x, cos_sin)?.reshape((g, dim))
    }
}

/// NeoX rotate-half RoPE over the leading `2·cos.dim(-1)` dims of each head of `x`
/// `[B, H, S, D]`, the rest passing through (nvidia/ops/qsa_pre_indexer.py:72-79).
fn rope(x: &Tensor, (cos, sin): &(Tensor, Tensor)) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let r = 2 * cos.dim(D::Minus1)?;
    let head = hanzo_nn::rotary_emb::rope(&x.narrow(D::Minus1, 0, r)?.contiguous()?, cos, sin)?;
    Tensor::cat(&[&head, &x.narrow(D::Minus1, r, d - r)?], D::Minus1)
}

/// Tokens each query attends, `[S, width]` i64: the tokens of its best `blocks` visible groups,
/// then its open tail group, ascending, `-1` past the end.
///
/// `q` `[S, heads, dim]` holds queries at positions `pos`; `kc` `[G, dim]` holds the sequence's
/// `G = len / ratio` compressed keys. A query at `p` sees group `j < min((p+1)/ratio, G)`
/// (nvidia/ops/qsa.py:65-68) and scores it `Σ_h ReLU(q_h·kc_j)/√dim` in f32 (:112-114). The
/// groups are ranked as the served kernel ranks them ([`top`]) and expanded with the tail
/// (:151-185).
pub(crate) fn select(q: &Tensor, kc: &Tensor, pos: &[usize], cfg: &Config) -> Result<Tensor> {
    let (rows, heads, dim) = q.dims3()?;
    let (groups, width) = (kc.dim(0)?, cfg.width());
    if (heads, dim) != (cfg.heads, cfg.dim) || kc.dims() != [groups, cfg.dim] {
        hanzo_ml::bail!(
            "QSA query {:?} or keys {:?} disagree with {cfg:?}",
            q.dims(),
            kc.dims()
        );
    }
    if pos.len() != rows {
        hanzo_ml::bail!("{} positions for {rows} queries", pos.len());
    }
    let qs = q.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let ks = kc.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let div = (dim as f64).sqrt() as f32;
    let mut out = vec![-1i64; rows * width];
    let mut score = Vec::with_capacity(groups);
    for ((row, qr), &p) in out
        .chunks_exact_mut(width)
        .zip(qs.chunks_exact(heads * dim))
        .zip(pos)
    {
        let visible = ((p + 1) / cfg.ratio).min(groups);
        score.clear();
        score.extend(ks.chunks_exact(dim).take(visible).map(|k| {
            qr.chunks_exact(dim)
                .map(|h| h.iter().zip(k).map(|(a, b)| a * b).sum::<f32>().max(0.0))
                .sum::<f32>()
                / div
        }));
        let tail = (p + 1) / cfg.ratio * cfg.ratio..=p;
        let tokens = top(&score, cfg.blocks)
            .into_iter()
            .flat_map(|j| j * cfg.ratio..(j + 1) * cfg.ratio)
            .chain(tail);
        for (cell, t) in row.iter_mut().zip(tokens) {
            *cell = t as i64;
        }
    }
    Tensor::from_vec(out, (rows, width), q.device())
}

/// Indices of the `k` best scores, ascending, as the served DET top-k picks them: value
/// descending, ties to the lower index, -0 tying +0 (kernel-det persistent_topk.cuh:49-57,
/// 119-135); every index when there are at most `k` (:1192-1197).
fn top(score: &[f32], k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..score.len()).collect();
    if k < idx.len() {
        let key = |i: usize| if score[i] == 0.0 { 0.0 } else { score[i] };
        idx.select_nth_unstable_by(k - 1, |&a, &b| key(b).total_cmp(&key(a)).then(a.cmp(&b)));
        idx.truncate(k);
        idx.sort_unstable();
    }
    idx
}

/// Paged-cache slots of a selection: token `t` sits at `table[t / size]·size + t % size`, the
/// engine's slot mapping (pipeline/inputs_processor.rs:1192-1210); `-1` stays `-1`. `table` is
/// one sequence's u32 block table.
pub(crate) fn slots(sel: &Tensor, table: &Tensor, size: usize) -> Result<Tensor> {
    let (rows, width) = sel.dims2()?;
    let table = table.to_vec1::<u32>()?;
    let out = sel
        .flatten_all()?
        .to_vec1::<i64>()?
        .into_iter()
        .map(|t| {
            if t < 0 {
                return Ok(-1);
            }
            let t = t as usize;
            match table.get(t / size) {
                Some(&b) => Ok((b as usize * size + t % size) as i64),
                None => hanzo_ml::bail!("token {t} lies past a {}-block table", table.len()),
            }
        })
        .collect::<Result<Vec<i64>>>()?;
    Tensor::from_vec(out, (rows, width), sel.device())
}

#[cfg(test)]
mod reference {
    //! Plain f64 transcriptions of the vLLM lines each function cites.

    /// Gemma RMS norm with its `1 + w` already in `w` (nvidia/ops/qsa_pre_indexer.py:68-71).
    pub(super) fn norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
        x.iter().zip(w).map(|(v, w)| v * r * w).collect()
    }

    /// NeoX rotate-half over the leading `rot` dims at `pos`, frequency `i` being
    /// `theta^(-2i/rot)` (nvidia/ops/qsa_pre_indexer.py:72-79).
    pub(super) fn rope(x: &mut [f64], pos: usize, rot: usize, theta: f64) {
        let half = rot / 2;
        for i in 0..half {
            let a = pos as f64 * theta.powf(-2.0 * i as f64 / rot as f64);
            let (x0, x1) = (x[i], x[i + half]);
            x[i] = x0 * a.cos() - x1 * a.sin();
            x[i + half] = x1 * a.cos() + x0 * a.sin();
        }
    }

    fn linear(x: &[f64], w: &[f64]) -> Vec<f64> {
        w.chunks(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Normed, roped queries `[heads·dim]` and the raw key of one token
    /// (nvidia/indexer_qsa.py:230-237, 276-282).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(
        u: &[f64],
        wq: &[f64],
        wk: &[f64],
        qn: &[f64],
        eps: f64,
        pos: usize,
        rot: usize,
        theta: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let q = linear(u, wq)
            .chunks(qn.len())
            .flat_map(|h| {
                let mut h = norm(h, qn, eps);
                rope(&mut h, pos, rot, theta);
                h
            })
            .collect();
        (q, linear(u, wk))
    }

    /// The group mean, rounded to bf16 (nvidia/ops/qsa_pre_indexer.py:261-269).
    pub(super) fn pool(raw: &[Vec<f64>]) -> Vec<f64> {
        (0..raw[0].len())
            .map(|d| {
                let mean = raw.iter().map(|k| k[d]).sum::<f64>() / raw.len() as f64;
                half::bf16::from_f64(mean).to_f64()
            })
            .collect()
    }

    /// Group scores of a query `[heads·dim]` at `p` over its visible compressed keys
    /// (nvidia/ops/qsa.py:65-68, 112-114).
    pub(super) fn scores(q: &[f64], kc: &[Vec<f64>], p: usize, ratio: usize) -> Vec<f64> {
        let dim = kc.first().map_or(1, Vec::len);
        kc.iter()
            .take((p + 1) / ratio)
            .map(|k| {
                q.chunks(dim)
                    .map(|h| h.iter().zip(k).map(|(a, b)| a * b).sum::<f64>().max(0.0))
                    .sum::<f64>()
                    / (dim as f64).sqrt()
            })
            .collect()
    }

    /// Brute-force selection of a query at `p`: a visible group is kept when fewer than `blocks`
    /// groups outrank it, outranking meaning a higher score or an equal one at a lower index
    /// (persistent_topk.cuh:49-57, 119-135); kept groups expand to their tokens, then the tail
    /// (nvidia/ops/qsa.py:151-185).
    pub(super) fn select(
        q: &[f64],
        kc: &[Vec<f64>],
        p: usize,
        ratio: usize,
        blocks: usize,
    ) -> Vec<i64> {
        let s = scores(q, kc, p, ratio);
        let mut row: Vec<i64> = Vec::new();
        for j in 0..s.len() {
            let above = (0..s.len())
                .filter(|&i| s[i] > s[j] || (s[i] == s[j] && i < j))
                .count();
            if above < blocks {
                row.extend((j * ratio..(j + 1) * ratio).map(|t| t as i64));
            }
        }
        row.extend(((p + 1) / ratio * ratio..=p).map(|t| t as i64));
        row.resize(blocks * ratio + ratio - 1, -1);
        row
    }

    /// Sparse GQA of one query `[heads·dim]` over the paged rows `slots`, `-1` skipped: softmax
    /// of `scale·q·k` over the valid rows, zero when there are none (nvidia/ops/qsa.py:257-323).
    /// `k` and `v` are `[slot, kv_heads, dim]`.
    pub(super) fn attend(
        q: &[f64],
        k: &[f64],
        v: &[f64],
        slots: &[i64],
        kv_heads: usize,
        dim: usize,
        scale: f64,
    ) -> Vec<f64> {
        let heads = q.len() / dim;
        let rows: Vec<usize> = slots
            .iter()
            .filter_map(|&s| usize::try_from(s).ok())
            .collect();
        let mut out = vec![0.0; heads * dim];
        for (h, (qh, oh)) in q.chunks(dim).zip(out.chunks_mut(dim)).enumerate() {
            let at = |t: usize| (t * kv_heads + h / (heads / kv_heads)) * dim;
            let logit: Vec<f64> = rows
                .iter()
                .map(|&t| scale * qh.iter().zip(&k[at(t)..]).map(|(a, b)| a * b).sum::<f64>())
                .collect();
            let max = logit.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = logit.iter().map(|l| (l - max).exp()).collect();
            let z: f64 = w.iter().sum();
            for (&t, w) in rows.iter().zip(&w) {
                for (o, x) in oh.iter_mut().zip(&v[at(t)..at(t) + dim]) {
                    *o += w / z * x;
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::repeat_kv;
    use crate::layers::Qwen3VLRotaryEmbedding;
    use hanzo_ml::quantized::{gguf_file, GgmlDType, QTensor};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::collections::HashMap;

    const HIDDEN: usize = 24;
    const ROT: usize = 8;
    const THETA: f32 = 1e7;
    const EPS: f32 = 1e-6;

    /// Multiples of `1/steps` in [-1, 1]. With a small `steps` they are bf16-exact and every dot
    /// product of them is exact in f32, so the f64 references rank the very same values.
    fn grid(rng: &mut StdRng, n: usize, steps: i32) -> Vec<f32> {
        (0..n)
            .map(|_| rng.random_range(-steps..=steps) as f32 / steps as f32)
            .collect()
    }

    fn wide(x: &[f32]) -> Vec<f64> {
        x.iter().map(|&v| f64::from(v)).collect()
    }

    fn f64s(t: &Tensor) -> Result<Vec<f64>> {
        t.to_dtype(DType::F64)?.flatten_all()?.to_vec1::<f64>()
    }

    /// Max error within `1e-5` of the reference's largest magnitude (at least 1).
    fn close(got: &Tensor, want: &[f64], what: &str) -> Result<()> {
        let got = f64s(got)?;
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(1f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(err <= 1e-5 * scale, "{what}: error {err} at scale {scale}");
        Ok(())
    }

    /// Each bf16 value is the reference rounded once: within half a bf16 ulp of it, which is
    /// `2^(⌊log2 |v|⌋ - 8)` as bf16 keeps 8 significant bits, plus the f32 arithmetic's own error.
    /// A second rounding costs up to another half ulp.
    fn rounded_once(got: &Tensor, want: &[f64], what: &str) -> Result<()> {
        assert_eq!(got.dtype(), DType::BF16, "{what}: dtype");
        let got = f64s(got)?;
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let half = f64::from_bits(w.abs().to_bits() & 0x7ff0_0000_0000_0000) / 256.0;
            assert!(
                (g - w).abs() <= half + 1e-6 * scale,
                "{what}[{i}] = {g}, reference {w}: more than one rounding"
            );
        }
        Ok(())
    }

    /// `[3, 1, S]` MRoPE position ids with the three planes equal, as for text.
    fn planes(pos: &[usize], dev: &Device) -> Result<Tensor> {
        let p: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
        Tensor::from_vec(p.repeat(3), (3, 1, pos.len()), dev)
    }

    /// Gemma norm weights as the GGUF stores them, f32 `1 + w`: `w` is a bf16 checkpoint value
    /// near -0.17, so most `1 + w` need more bits than bf16 keeps.
    fn folded(rng: &mut StdRng, n: usize) -> Vec<f32> {
        (0..n)
            .map(|_| 1.0 + half::bf16::from_f32(rng.random_range(-0.3..-0.05)).to_f32())
            .collect()
    }

    struct Weights {
        q: Vec<f32>,
        k: Vec<f32>,
        q_norm: Vec<f32>,
        k_norm: Vec<f32>,
    }

    /// A GGUF holding one QSA layer's indexer keys and `blk.3.indexer.*` tensors, projections in
    /// BF16 as the real file stores them. `Content` admits only architectures it knows, so the
    /// file declares `qwen35moe`; the indexer reads its keys under `qwen4exp` regardless.
    fn fixture(path: &std::path::Path, cfg: &Config, rng: &mut StdRng) -> Result<Weights> {
        let w = Weights {
            q: grid(rng, cfg.heads * cfg.dim * HIDDEN, 16),
            k: grid(rng, cfg.dim * HIDDEN, 16),
            q_norm: folded(rng, cfg.dim),
            k_norm: folded(rng, cfg.dim),
        };
        let dev = Device::Cpu;
        let q = QTensor::quantize(
            &Tensor::from_vec(w.q.clone(), (cfg.heads * cfg.dim, HIDDEN), &dev)?,
            GgmlDType::BF16,
        )?;
        let k = QTensor::quantize(
            &Tensor::from_vec(w.k.clone(), (cfg.dim, HIDDEN), &dev)?,
            GgmlDType::BF16,
        )?;
        let qn = QTensor::quantize(
            &Tensor::from_vec(w.q_norm.clone(), cfg.dim, &dev)?,
            GgmlDType::F32,
        )?;
        let kn = QTensor::quantize(
            &Tensor::from_vec(w.k_norm.clone(), cfg.dim, &dev)?,
            GgmlDType::F32,
        )?;
        let tensors = [
            ("blk.3.indexer.q_proj.weight", &q),
            ("blk.3.indexer.k_proj.weight", &k),
            ("blk.3.indexer.q_norm.weight", &qn),
            ("blk.3.indexer.k_norm.weight", &kn),
        ];
        let u32v = |v: usize| gguf_file::Value::U32(v as u32);
        let ratios = gguf_file::Value::Array(
            [0, 0, 0, cfg.ratio as i32]
                .map(gguf_file::Value::I32)
                .to_vec(),
        );
        let arch = gguf_file::Value::String("qwen35moe".to_string());
        let (heads, dim, top_k) = (u32v(cfg.heads), u32v(cfg.dim), u32v(cfg.blocks * cfg.ratio));
        let metadata = [
            ("general.architecture", &arch),
            ("qwen4exp.attention.indexer.head_count", &heads),
            ("qwen4exp.attention.indexer.key_length", &dim),
            ("qwen4exp.attention.indexer.top_k", &top_k),
            ("qwen4exp.attention.compress_ratios", &ratios),
        ];
        let mut file = std::fs::File::create(path).map_err(hanzo_ml::Error::msg)?;
        gguf_file::write(&mut file, &metadata, &tensors)?;
        Ok(w)
    }

    /// Layer 3's config and indexer, read back from the GGUF at `path`.
    fn load(path: &std::path::Path) -> Result<(Config, Indexer)> {
        let mut files = [std::fs::File::open(path).map_err(hanzo_ml::Error::msg)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let cfg = Config::from_gguf(
            &ContentMetadata {
                path_prefix: "qwen4exp",
                metadata: ct.get_metadata(),
            },
            3,
        )?
        .expect("layer 3 has an indexer");
        let indexer = Indexer::from_gguf(&mut ct, "blk.3", cfg, EPS, &Device::Cpu)?;
        Ok((cfg, indexer))
    }

    /// The production header's values give 512 groups and a width of 2051; a GDN layer, whose
    /// ratio is 0, and the MTP layer, past the table, have no indexer.
    #[test]
    fn config_reads_gguf_keys() -> Result<()> {
        use gguf_file::Value;
        let ratios = (0..48)
            .map(|l| Value::I32(if l % 4 == 3 { 4 } else { 0 }))
            .collect();
        let metadata: HashMap<String, Value> = [
            ("indexer.head_count", Value::U32(4)),
            ("indexer.key_length", Value::U32(128)),
            ("indexer.top_k", Value::U32(2048)),
            ("compress_ratios", Value::Array(ratios)),
        ]
        .into_iter()
        .map(|(k, v)| (format!("qwen4exp.attention.{k}"), v))
        .collect();
        let md = ContentMetadata {
            path_prefix: "qwen4exp",
            metadata: &metadata,
        };
        let cfg = Config::from_gguf(&md, 47)?;
        assert_eq!(
            cfg,
            Some(Config {
                heads: 4,
                dim: 128,
                ratio: 4,
                blocks: 512
            })
        );
        assert_eq!(cfg.map(|c| c.width()), Some(2051));
        assert_eq!(Config::from_gguf(&md, 46)?, None);
        assert_eq!(Config::from_gguf(&md, 48)?, None);
        Ok(())
    }

    /// `project` and `compress`, loaded from a GGUF, match the f64 references. The raw keys sit
    /// on a 1/4096 grid, so the f32 mean is exact and only the bf16 rounding separates it from the
    /// f64 mean; skipping that rounding misses the reference by far more than the tolerance.
    #[test]
    fn indexer_matches_reference() -> Result<()> {
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(0x7173_6101);
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let path = dir.path().join("qsa.gguf");
        let tiny = Config {
            heads: 2,
            dim: 16,
            ratio: 4,
            blocks: 2,
        };
        let w = fixture(&path, &tiny, &mut rng)?;
        let (cfg, indexer) = load(&path)?;
        assert_eq!((cfg, cfg.width()), (tiny, 11));
        let rotary = Qwen3VLRotaryEmbedding::new(THETA, ROT, &dev, vec![2, 1, 1])?;

        // A chunk at positions 5..16.
        let pos: Vec<usize> = (5..16).collect();
        let u = grid(&mut rng, pos.len() * HIDDEN, 16);
        let (q, k) = indexer.project(
            &Tensor::from_vec(u.clone(), (1, pos.len(), HIDDEN), &dev)?,
            &rotary.compute_cos_sin(&planes(&pos, &dev)?, DType::F32)?,
        )?;
        let (mut want_q, mut want_k) = (Vec::new(), Vec::new());
        for (x, &p) in u.chunks(HIDDEN).zip(&pos) {
            let (q, k) = reference::project(
                &wide(x),
                &wide(&w.q),
                &wide(&w.k),
                &wide(&w.q_norm),
                f64::from(EPS),
                p,
                ROT,
                f64::from(THETA),
            );
            want_q.extend(q);
            want_k.extend(k);
        }
        assert_eq!(q.dims(), &[1, pos.len(), cfg.heads, cfg.dim]);
        close(&q, &want_q, "queries")?;
        close(&k, &want_k, "raw keys")?;

        // Groups 3, 4 and 5, roped at their first positions 12, 16 and 20.
        let groups = [3usize, 4, 5];
        let raw = grid(&mut rng, groups.len() * cfg.ratio * cfg.dim, 4096);
        let first: Vec<usize> = groups.iter().map(|j| j * cfg.ratio).collect();
        let kc = indexer.compress(
            &Tensor::from_vec(raw.clone(), (groups.len() * cfg.ratio, cfg.dim), &dev)?,
            &rotary.compute_cos_sin(&planes(&first, &dev)?, DType::F32)?,
        )?;
        let (mut want, mut plain) = (Vec::new(), Vec::new());
        for (keys, &p) in raw.chunks(cfg.ratio * cfg.dim).zip(&first) {
            let keys: Vec<Vec<f64>> = keys.chunks(cfg.dim).map(wide).collect();
            let mean: Vec<f64> = (0..cfg.dim)
                .map(|d| keys.iter().map(|k| k[d]).sum::<f64>() / cfg.ratio as f64)
                .collect();
            for (pooled, out) in [(reference::pool(&keys), &mut want), (mean, &mut plain)] {
                let mut y = reference::norm(&pooled, &wide(&w.k_norm), f64::from(EPS));
                reference::rope(&mut y, p, ROT, f64::from(THETA));
                out.extend(y);
            }
        }
        assert_eq!(kc.dims(), &[groups.len(), cfg.dim]);
        close(&kc, &want, "compressed keys")?;
        let got = kc.flatten_all()?.to_vec1::<f32>()?;
        let miss = got
            .iter()
            .zip(&plain)
            .fold(0f64, |m, (a, b)| m.max((f64::from(*a) - b).abs()));
        assert!(miss > 1e-4, "the bf16 rounding left no trace ({miss})");
        Ok(())
    }

    /// With bf16 activations each norm keeps x, its rms and the f32 weight `1 + w` in f32 and
    /// rounds once, before RoPE (nvidia/ops/qsa_pre_indexer.py:68-71; the query at :174-186, the
    /// compressed key at :322-334). RoPE at position 0 is the identity, so each output is that one
    /// rounding of the f64 norm of the bf16 values the indexer normed. Rounding `1 + w` to bf16
    /// first moves many outputs an ulp. The bound is half a bf16 ulp, not 1e-5, because the bf16
    /// rounding is what is under test.
    #[test]
    fn norms_round_once() -> Result<()> {
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(0x7173_6105);
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let path = dir.path().join("qsa.gguf");
        let tiny = Config {
            heads: 2,
            dim: 16,
            ratio: 4,
            blocks: 2,
        };
        let w = fixture(&path, &tiny, &mut rng)?;
        let (cfg, indexer) = load(&path)?;
        let rotary = Qwen3VLRotaryEmbedding::new(THETA, ROT, &dev, vec![2, 1, 1])?;
        let origin = |n: usize| rotary.compute_cos_sin(&planes(&vec![0; n], &dev)?, DType::BF16);

        let s = 11;
        let u = Tensor::from_vec(grid(&mut rng, s * HIDDEN, 16), (1, s, HIDDEN), &dev)?
            .to_dtype(DType::BF16)?;
        let (q, _) = indexer.project(&u, &origin(s)?)?;
        let want: Vec<f64> = f64s(&indexer.q.forward(&u)?)?
            .chunks(cfg.dim)
            .flat_map(|h| reference::norm(h, &wide(&w.q_norm), f64::from(EPS)))
            .collect();
        rounded_once(&q, &want, "queries")?;

        let groups = 3;
        let raw = Tensor::from_vec(
            grid(&mut rng, groups * cfg.ratio * cfg.dim, 4096),
            (groups * cfg.ratio, cfg.dim),
            &dev,
        )?
        .to_dtype(DType::BF16)?;
        let kc = indexer.compress(&raw, &origin(groups)?)?;
        let want: Vec<f64> = f64s(&raw)?
            .chunks(cfg.ratio * cfg.dim)
            .flat_map(|keys| {
                let keys: Vec<Vec<f64>> = keys.chunks(cfg.dim).map(<[f64]>::to_vec).collect();
                reference::norm(&reference::pool(&keys), &wide(&w.k_norm), f64::from(EPS))
            })
            .collect();
        rounded_once(&kc, &want, "compressed keys")
    }

    /// Value descending, ties to the lower index, -0 tying +0, emitted ascending. Ordering -0
    /// below +0, as a bare `total_cmp` does, would pick index 2 over index 1 at `k = 4`.
    #[test]
    fn top_follows_det_rule() {
        let s = [0.5, -0.0, 0.0, 0.5, -0.0, 1.0, 0.0];
        assert_eq!(top(&s, 2), vec![0, 5]);
        assert_eq!(top(&s, 4), vec![0, 1, 3, 5]);
        assert_eq!(top(&s, 5), vec![0, 1, 2, 3, 5]);
        assert_eq!(top(&s, 7), (0..7).collect::<Vec<_>>());
        assert_eq!(top(&s, 9), (0..7).collect::<Vec<_>>());
    }

    /// Every query of a 61-token sequence against the brute-force reference. Zero keys score
    /// exactly 0 against every query and copied keys tie exactly with their source, so many rows
    /// cut the ranking inside a run of equal scores, zeros and non-zeros both.
    #[test]
    fn select_matches_brute_force() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config {
            heads: 2,
            dim: 16,
            ratio: 4,
            blocks: 3,
        };
        let (len, groups, d) = (61, 15, cfg.dim);
        let mut rng = StdRng::seed_from_u64(0x7173_6102);
        let q = grid(&mut rng, len * cfg.heads * d, 4);
        let mut kc = grid(&mut rng, groups * d, 4);
        for j in [1, 4, 6, 10, 13] {
            kc[j * d..(j + 1) * d].fill(0.0);
        }
        for (to, from) in [(9, 2), (12, 5), (14, 7)] {
            kc.copy_within(from * d..(from + 1) * d, to * d);
        }
        let pos: Vec<usize> = (0..len).collect();
        let sel = select(
            &Tensor::from_vec(q.clone(), (len, cfg.heads, d), &dev)?,
            &Tensor::from_vec(kc.clone(), (groups, d), &dev)?,
            &pos,
            &cfg,
        )?
        .to_vec2::<i64>()?;
        let keys: Vec<Vec<f64>> = kc.chunks(d).map(wide).collect();
        let (mut ties, mut zeros) = (0, 0);
        for ((row, x), &p) in sel.iter().zip(q.chunks(cfg.heads * d)).zip(&pos) {
            let x = wide(x);
            assert_eq!(
                *row,
                reference::select(&x, &keys, p, cfg.ratio, cfg.blocks),
                "query at {p}"
            );
            let mut s = reference::scores(&x, &keys, p, cfg.ratio);
            s.sort_by(|a, b| b.total_cmp(a));
            if s.len() > cfg.blocks && s[cfg.blocks - 1] == s[cfg.blocks] {
                ties += 1;
                zeros += usize::from(s[cfg.blocks] == 0.0);
            }
        }
        assert!(
            zeros > 0 && ties > zeros,
            "cuts inside ties: {ties}, at zero: {zeros}"
        );
        Ok(())
    }

    /// Up to `width` tokens every query attends every position up to its own; one token more and
    /// the last query drops a group.
    #[test]
    fn select_is_dense_within_budget() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config {
            heads: 2,
            dim: 16,
            ratio: 4,
            blocks: 3,
        };
        let width = cfg.width();
        let mut rng = StdRng::seed_from_u64(0x7173_6103);
        for len in 1..=width + 1 {
            let groups = len / cfg.ratio;
            let pos: Vec<usize> = (0..len).collect();
            let sel = select(
                &Tensor::from_vec(
                    grid(&mut rng, len * cfg.heads * cfg.dim, 64),
                    (len, cfg.heads, cfg.dim),
                    &dev,
                )?,
                &Tensor::from_vec(
                    grid(&mut rng, groups * cfg.dim, 64),
                    (groups, cfg.dim),
                    &dev,
                )?,
                &pos,
                &cfg,
            )?
            .to_vec2::<i64>()?;
            for (p, row) in sel.iter().enumerate() {
                let dense: Vec<i64> = (0..width as i64)
                    .map(|t| if t <= p as i64 { t } else { -1 })
                    .collect();
                if p < width {
                    assert_eq!(*row, dense, "len {len}, query at {p}");
                } else {
                    let kept = row.iter().filter(|&&t| t >= 0).count();
                    assert_eq!(kept, cfg.blocks * cfg.ratio, "len {len}, query at {p}");
                }
            }
        }
        Ok(())
    }

    /// Attention over the selected tokens, gathered from a paged cache through `slots`, equals
    /// dense attention masked to the same tokens, and differs from plain causal attention.
    #[test]
    fn sparse_attention_equals_masked_dense() -> Result<()> {
        let dev = Device::Cpu;
        let cfg = Config {
            heads: 2,
            dim: 16,
            ratio: 4,
            blocks: 3,
        };
        let (heads, kv_heads, hd, len, size) = (4usize, 2usize, 8usize, 40usize, 8usize);
        let table = [6u32, 2, 9, 0, 4];
        let pos = [0usize, 2, 3, 7, 11, 15, 16, 22, 27, 31, 35, 36, 37, 38, 39];
        let mut rng = StdRng::seed_from_u64(0x7173_6104);
        let mut uniform =
            |n: usize| -> Vec<f32> { (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect() };
        let (q, k, v) = (
            uniform(pos.len() * heads * hd),
            uniform(len * kv_heads * hd),
            uniform(len * kv_heads * hd),
        );
        let (iq, kc) = (
            uniform(pos.len() * cfg.heads * cfg.dim),
            uniform(len / cfg.ratio * cfg.dim),
        );
        let sel = select(
            &Tensor::from_vec(iq, (pos.len(), cfg.heads, cfg.dim), &dev)?,
            &Tensor::from_vec(kc, (len / cfg.ratio, cfg.dim), &dev)?,
            &pos,
            &cfg,
        )?;
        let at = slots(&sel, &Tensor::new(&table, &dev)?, size)?.to_vec2::<i64>()?;
        let sel = sel.to_vec2::<i64>()?;
        for (a, s) in at.iter().flatten().zip(sel.iter().flatten()) {
            assert_eq!(*a < 0, *s < 0, "slot {a} for token {s}");
        }
        assert!(
            sel.iter()
                .zip(&pos)
                .any(|(r, &p)| r.iter().filter(|&&t| t >= 0).count() <= p),
            "no query dropped a token"
        );

        // The paged cache: token t's row at slot table[t / size]·size + t % size.
        let row = kv_heads * hd;
        let (mut pk, mut pv) = (vec![0f64; 10 * size * row], vec![0f64; 10 * size * row]);
        for t in 0..len {
            let s = table[t / size] as usize * size + t % size;
            pk[s * row..(s + 1) * row].copy_from_slice(&wide(&k[t * row..(t + 1) * row]));
            pv[s * row..(s + 1) * row].copy_from_slice(&wide(&v[t * row..(t + 1) * row]));
        }
        let scale = 1.0 / (hd as f64).sqrt();
        let sparse: Vec<f64> = q
            .chunks(heads * hd)
            .zip(&at)
            .flat_map(|(x, a)| reference::attend(&wide(x), &pk, &pv, a, kv_heads, hd, scale))
            .collect();

        let masks = |keep: &dyn Fn(usize, usize) -> bool| -> Result<Tensor> {
            let m: Vec<f32> = (0..pos.len())
                .flat_map(|s| (0..len).map(move |t| (s, t)))
                .map(|(s, t)| if keep(s, t) { 0.0 } else { f32::NEG_INFINITY })
                .collect();
            Tensor::from_vec(m, (pos.len(), len), &dev)
        };
        // Dense masked attention in f32 tensor ops. The engine's CPU `Sdpa` multiplies through f16
        // (hanzo-quant `MatMul::matmul`), which alone misses by ~1e-3.
        let dense = |mask: Tensor| -> Result<Tensor> {
            let q = Tensor::from_vec(q.clone(), (1, pos.len(), heads, hd), &dev)?
                .transpose(1, 2)?
                .contiguous()?;
            let kv = |x: &[f32]| -> Result<Tensor> {
                repeat_kv(
                    Tensor::from_vec(x.to_vec(), (1, len, kv_heads, hd), &dev)?
                        .transpose(1, 2)?
                        .contiguous()?,
                    heads / kv_heads,
                )
            };
            let logits = (q.matmul(&kv(&k)?.t()?)? * scale)?.broadcast_add(&mask)?;
            hanzo_nn::ops::softmax_last_dim(&logits)?
                .matmul(&kv(&v)?)?
                .transpose(1, 2)
        };
        let masked = dense(masks(&|s, t| sel[s].contains(&(t as i64)))?)?;
        close(&masked, &sparse, "sparse attention")?;

        let causal = dense(masks(&|s, t| t <= pos[s])?)?;
        let gap = (causal - masked)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(gap > 1e-3, "the selection changed nothing ({gap})");
        Ok(())
    }
}
