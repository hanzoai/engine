#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! zen6's QSA: attention restricted to the best compressed key groups.
//!
//! A weight-free indexer projects `heads` query heads and one key head of width `dim` from the
//! block input. Each run of `ratio` raw keys pools into one compressed key. A query scores every
//! visible group by `Σ_h ReLU(q_h·kc_j)/√dim`, keeps the best `blocks`, and attends their tokens
//! plus its own open tail group: at most `width = blocks·ratio + ratio - 1` tokens, so QSA is
//! exactly dense attention for sequences of up to `width` tokens.
//!
//! # The index
//!
//! Beside its K/V, each attention layer keeps an index entry of two rows per token, `dim` wide
//! and one head: in K the compressed key of group `j` at the group's last token
//! `ratio·j + ratio - 1` and zero at every other token, in V the token's raw key. Rows per token
//! give the index the K/V's own slots, so everything the engine does to a sequence's cache
//! (paging, prefix reuse, trimming a rejected draft, swapping a sequence out) keeps the two
//! consistent with no bookkeeping of QSA's own. A chunk closes the groups that end inside it from
//! its own raw keys and the at most `ratio - 1` raw keys before it, read back from the index.
//!
//! # The forward
//!
//! Every forward projects the indexer and writes the index. While no query of the forward sees
//! more than `blocks` groups every query keeps every token up to its own, so the layer runs its
//! dense attention unchanged. Past that, each sequence
//!
//! 1. reads its `G = len / ratio` compressed keys from the index;
//! 2. scores its queries against them on the device, as many queries at a time as keep the
//!    scores within [`SCORE_BYTES`], and ranks each query's groups on the host by the served
//!    kernel's rule ([`select`]);
//! 3. groups consecutive queries while attending the union of their tokens under a mask spends
//!    at most twice the scores their own selections need, gathers each union's K/V rows once,
//!    and runs the QSA kernel's arithmetic ([`softmax_pv`]) over it under a mask, grouped-query
//!    heads folded into rows ([`sparse`]).
//!
//! Per layer and sequence, `n` queries at context `T` cost `O(n·T/ratio·heads·dim)` to score and
//! `O(n·width)` rows of attention, against `O(n·T)` for dense attention; a decode step reads the
//! `T/ratio` compressed keys and `width` K/V rows. The transients are bounded independently of
//! `T` except the compressed keys themselves: [`SCORE_BYTES`] of scores, [`UNION`] gathered
//! rows, and [`LOGITS`] mask entries.
//!
//! The top-k is the deterministic one: score descending, ties to the lower index.

use std::io::{Read, Seek};
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::Linear;
use hanzo_quant::{QuantMethod, QuantMethodConfig, ShardedVarBuilder, UnquantLinear};
use rayon::prelude::*;

use crate::attention::AttentionMask;
use crate::gguf::Content;
use crate::layers::QRmsNorm;
use crate::models::quantized_qwen3_5_moe::{gguf_qmm, GatedFullAttention};
use crate::paged_attention::AttentionBackendKind;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::KvCache;
use crate::utils::gguf_metadata::ContentMetadata;

/// Bytes of f32 scores one chunk of queries may hold, with their per-head terms: the served
/// kernel's logits workspace, 128 MiB.
const SCORE_BYTES: usize = 128 << 20;

/// Tokens whose K/V rows one group of queries gathers.
const UNION: usize = 1 << 15;

/// Mask entries, query rows times gathered tokens, of one group of queries.
const LOGITS: usize = 1 << 20;

/// Attention with the QSA kernel's math over dense causal keys, which is what QSA computes while
/// every visible key is selected (one tile), by [`softmax_pv`].
///
/// `q` `[b, hq, s, d]`, `k`/`v` `[b, hkv, L, d]` with the `s` queries at positions `L - s ..
/// L`; returns `[b, hq, s, d]` in `q`'s dtype.
pub(crate) fn attend(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
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
    let offset = l - s;
    let mask: Vec<f32> = (0..s)
        .flat_map(|i| {
            (0..l).map(move |j| {
                if j <= offset + i {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    let mask = Tensor::from_vec(mask, (1, 1, s, l), q.device())?;
    softmax_pv(q, &k, &v, &mask)
}

/// The QSA kernel's attention arithmetic: f32 scores scaled by
/// `d^-0.5 · log2 e` plus the additive `mask`, `exp2` against the row max, the probabilities
/// rounded to the activation dtype for the PV product while the normalizer sums them unrounded,
/// an f32 PV, one division and one rounding. `q` `[.., m, d]`, `k`/`v` `[.., u, d]`, `mask`
/// broadcast to `[.., m, u]`; returns `[.., m, d]` in `q`'s dtype.
fn softmax_pv(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let dtype = q.dtype();
    let d = q.dim(D::Minus1)?;
    let (k, v) = (k.to_dtype(DType::F32)?, v.to_dtype(DType::F32)?);
    let scale = ((d as f64).powf(-0.5) * std::f64::consts::LOG2_E) as f32;
    let scores = (q.to_dtype(DType::F32)?.matmul(&k.t()?.contiguous()?)? * f64::from(scale))?;
    let scores = scores.broadcast_add(mask)?;
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
    /// the MTP layer past the table). `top_k` counts tokens, so `blocks = top_k / ratio`.
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
        Self::new(
            get("attention.indexer.head_count")?,
            get("attention.indexer.key_length")?,
            ratio,
            get("attention.indexer.top_k")?,
        )
        .map(Some)
        .map_err(|e| hanzo_ml::Error::Msg(format!("layer {layer}: {e}")))
    }

    /// `heads` query heads `dim` wide keeping `tokens` per query in groups of `ratio`.
    pub(crate) fn new(heads: usize, dim: usize, ratio: usize, tokens: usize) -> Result<Self> {
        // Every value positive and the budget a whole number of groups.
        if heads == 0 || dim == 0 || ratio == 0 || tokens == 0 || !tokens.is_multiple_of(ratio) {
            hanzo_ml::bail!("QSA heads {heads}, dim {dim}, top_k {tokens}, ratio {ratio}");
        }
        Ok(Self {
            heads,
            dim,
            ratio,
            blocks: tokens / ratio,
        })
    }

    /// Selection width: `blocks` whole groups plus a tail of at most `ratio - 1` tokens.
    pub(crate) fn width(&self) -> usize {
        self.blocks * self.ratio + self.ratio - 1
    }

    /// Groups a query at `p` sees: those complete by `p`, `(p + 1) / ratio`.
    fn visible(&self, p: usize) -> usize {
        (p + 1) / self.ratio
    }
}

/// One layer's indexer. The q/k norms are Gemma norms: the GGUF carries their `1 + w` and the
/// safetensors checkpoint `w`, and either way they run as plain RMS norms, in f32 with `1 + w`.
pub(crate) struct Indexer {
    q: Arc<dyn QuantMethod>,
    k: Arc<dyn QuantMethod>,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    cfg: Config,
}

impl Indexer {
    /// `{prefix}.indexer.{q_proj, k_proj, q_norm, k_norm}`: the checkpoint's `index_qk_proj`
    /// split into its query rows and its key rows.
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

    /// `{vb}.{index_qk_proj, q_layernorm, k_layernorm}` of a safetensors checkpoint
    /// (`...self_attn.indexer`): the projection's first `heads·dim` rows are the queries and the
    /// next `dim` the key, which the GGUF stores as `q_proj` and `k_proj`; the norms are `w`, held
    /// as `1 + w` in f32.
    pub(crate) fn new(
        vb: &ShardedVarBuilder,
        cfg: Config,
        hidden: usize,
        eps: f64,
    ) -> Result<Self> {
        let rows = cfg.heads * cfg.dim;
        let qk = vb.get((rows + cfg.dim, hidden), "index_qk_proj.weight")?;
        let linear = |w: Tensor| -> Result<Arc<dyn QuantMethod>> {
            Ok(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(w, None)),
            )?))
        };
        let norm = |name: &str| -> Result<QRmsNorm> {
            let w = vb.get_with_hints_dtype(
                (cfg.dim,),
                &format!("{name}.weight"),
                Default::default(),
                DType::F32,
            )?;
            Ok(QRmsNorm::from_weight((w + 1.0)?, eps))
        };
        Ok(Self {
            q: linear(qk.narrow(0, 0, rows)?)?,
            k: linear(qk.narrow(0, rows, cfg.dim)?)?,
            q_norm: norm("q_layernorm")?,
            k_norm: norm("k_layernorm")?,
            cfg,
        })
    }

    /// Queries `[B, S, heads, dim]`, normed in f32 and rounded once, then roped, and raw keys
    /// `[B, S, dim]` of `u` `[B, S, hidden]`; `cos_sin` holds the rotary tables at the chunk's
    /// positions.
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
    /// the group's first position `ratio·j`, where `cos_sin` holds one row per group.
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

    /// The index's K rows for a chunk, `[rows, dim]`: the compressed key of each group that closes
    /// inside the chunk at the group's last token, zero elsewhere. `raw` holds the chunk's first
    /// `n` raw keys, at positions `a..a + n`, and `prior` the `a % ratio` raw keys before them,
    /// the open group's head, which only a chunk that closes a group reads. The groups are roped
    /// by `attn`'s rotary at their first positions, as the served model ropes them with the main
    /// attention's.
    fn closed(
        &self,
        attn: &GatedFullAttention,
        raw: &Tensor,
        prior: Option<&Tensor>,
        a: usize,
        rows: usize,
    ) -> Result<Tensor> {
        let (n, dim) = raw.dims2()?;
        let r = self.cfg.ratio;
        let p = a % r;
        let zeros = |len: usize| Tensor::zeros((len, dim), raw.dtype(), raw.device());
        let groups = (p + n) / r;
        if groups == 0 {
            return zeros(rows);
        }
        if prior.map_or(0, |t| t.dim(0).unwrap_or(0)) != p {
            hanzo_ml::bail!("QSA: a chunk at {a} needs the {p} raw keys before it");
        }
        let keys = match prior {
            Some(prior) => Tensor::cat(&[prior, raw], 0)?,
            None => raw.clone(),
        };
        let first: Vec<u32> = (0..groups).map(|j| (a - p + j * r) as u32).collect();
        let planes = Tensor::from_vec(first.repeat(3), (3, 1, groups), raw.device())?;
        let kc = self.compress(
            &keys.narrow(0, 0, groups * r)?,
            &attn.rotary_cos_sin(&planes, raw.dtype())?,
        )?;
        // Each group as `ratio` rows with its key in the last, less the rows before the chunk.
        let placed = Tensor::cat(
            &[
                &Tensor::zeros((groups, r - 1, dim), kc.dtype(), kc.device())?,
                &kc.unsqueeze(1)?,
            ],
            1,
        )?
        .reshape((groups * r, dim))?
        .narrow(0, p, groups * r - p)?;
        Tensor::cat(&[&placed, &zeros(rows - (groups * r - p))?], 0)
    }
}

/// NeoX rotate-half RoPE over the leading `2·cos.dim(-1)` dims of each head of `x`
/// `[B, H, S, D]`, the rest passing through.
fn rope(x: &Tensor, (cos, sin): &(Tensor, Tensor)) -> Result<Tensor> {
    let d = x.dim(D::Minus1)?;
    let r = 2 * cos.dim(D::Minus1)?;
    let head = hanzo_nn::rotary_emb::rope(&x.narrow(D::Minus1, 0, r)?.contiguous()?, cos, sin)?;
    Tensor::cat(&[&head, &x.narrow(D::Minus1, r, d - r)?], D::Minus1)
}

/// `Σ_h ReLU(q_h·kc_j)/√dim` of queries `q` `[rows, heads, dim]` against compressed keys `kc`
/// `[G, dim]`, as `[rows, G]` f32 on their device. Products of bf16 values are exact in f32, so
/// this is the served kernel's arithmetic up to the order of its sums.
fn scores(q: &Tensor, kc: &Tensor) -> Result<Tensor> {
    let (rows, heads, dim) = q.dims3()?;
    let g = kc.dim(0)?;
    let q = q
        .to_dtype(DType::F32)?
        .reshape((rows * heads, dim))?
        .t()?
        .contiguous()?;
    let s = kc
        .to_dtype(DType::F32)?
        .matmul(&q)?
        .relu()?
        .reshape((g, rows, heads))?
        .sum(2)?;
    s.t()?.contiguous()? / (dim as f64).sqrt()
}

/// Tokens each query attends, ascending: the tokens of its best `blocks` visible groups, then its
/// open tail group.
///
/// `q` `[S, heads, dim]` holds queries at positions `pos`; `kc` `[G, dim]` holds the sequence's
/// `G = len / ratio` compressed keys. A query at `p` sees group `j < min((p+1)/ratio, G)` and
/// scores it by [`scores`]. The groups are ranked as the served kernel ranks them ([`top`]) and
/// expanded with the tail. Only queries that must drop a group are scored, as many at a time as
/// keep their scores within [`SCORE_BYTES`].
pub(crate) fn select(
    q: &Tensor,
    kc: &Tensor,
    pos: &[usize],
    cfg: &Config,
) -> Result<Vec<Vec<u32>>> {
    let (rows, heads, dim) = q.dims3()?;
    let groups = kc.dim(0)?;
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
    let q = q.contiguous()?;
    let visible = |p: usize| cfg.visible(p).min(groups);
    let expand = |kept: Vec<usize>, p: usize| -> Vec<u32> {
        let tail = cfg.visible(p) * cfg.ratio..=p;
        kept.into_iter()
            .flat_map(|j| j * cfg.ratio..(j + 1) * cfg.ratio)
            .chain(tail)
            .map(|t| t as u32)
            .collect()
    };
    let mut out: Vec<Vec<u32>> = pos
        .iter()
        .map(|&p| {
            if visible(p) <= cfg.blocks {
                expand((0..visible(p)).collect(), p)
            } else {
                Vec::new()
            }
        })
        .collect();
    let ranked: Vec<usize> = (0..rows)
        .filter(|&i| visible(pos[i]) > cfg.blocks)
        .collect();
    let Some(span) = ranked.iter().map(|&i| visible(pos[i])).max() else {
        return Ok(out);
    };
    let per = (SCORE_BYTES / (4 * (2 * heads + 1) * span)).max(1);
    for chunk in ranked.chunks(per) {
        let cols = chunk.iter().map(|&i| visible(pos[i])).max().unwrap_or(0);
        let ids: Vec<u32> = chunk.iter().map(|&i| i as u32).collect();
        let ids = Tensor::from_vec(ids, chunk.len(), q.device())?;
        let s = scores(&q.index_select(&ids, 0)?, &kc.narrow(0, 0, cols)?)?.to_vec2::<f32>()?;
        let rows: Vec<Vec<u32>> = chunk
            .par_iter()
            .zip(s.par_iter())
            .map(|(&i, row)| expand(top(&row[..visible(pos[i])], cfg.blocks), pos[i]))
            .collect();
        for (&i, row) in chunk.iter().zip(rows) {
            out[i] = row;
        }
    }
    Ok(out)
}

/// Indices of the `k` best scores, ascending, as the served DET top-k picks them: value
/// descending, ties to the lower index, -0 tying +0; every index when there are at most `k`.
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

/// How a paged cache lays out one entry (hanzo-paged-attn `reshape_and_cache`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Layout {
    /// `[blocks, heads, dim / x, size, x]`, the standard key cache.
    Key,
    /// `[blocks, heads, dim, size]`, the standard value cache.
    Value,
    /// `[blocks, heads, size, dim]`, FlashInfer's key and value cache.
    Hnd,
}

impl Layout {
    /// The layouts of a paged `(K, V)` pair, which QSA reads only unquantized.
    fn of(k: &Tensor, v: &Tensor) -> Result<(Self, Self)> {
        if !matches!(k.dtype(), DType::BF16 | DType::F16 | DType::F32) || k.dtype() != v.dtype() {
            hanzo_ml::bail!(
                "QSA reads an unquantized paged KV cache, got {:?}/{:?}",
                k.dtype(),
                v.dtype()
            );
        }
        Ok(match AttentionBackendKind::from_cache(k, v) {
            AttentionBackendKind::Standard => (Self::Key, Self::Value),
            AttentionBackendKind::FlashInfer => (Self::Hnd, Self::Hnd),
        })
    }

    /// `(heads, dim, run, per_block, starts)` of a cache: a token's `dim` values for one head sit
    /// in runs of `run` contiguous elements, and viewing the cache as rows of `run` elements, the
    /// token at offset `o` of block `b` has its runs at `b·per_block + o + starts[i]`, head-major.
    fn runs(self, cache: &Tensor) -> Result<(usize, usize, usize, usize, Vec<u32>)> {
        Ok(match self {
            Self::Key => {
                let (_, heads, chunks, size, x) = cache.dims5()?;
                let starts = (0..heads * chunks).map(|c| (c * size) as u32).collect();
                (heads, chunks * x, x, heads * chunks * size, starts)
            }
            Self::Value => {
                let (_, heads, dim, size) = cache.dims4()?;
                let starts = (0..heads * dim).map(|c| (c * size) as u32).collect();
                (heads, dim, 1, heads * dim * size, starts)
            }
            Self::Hnd => {
                let (_, heads, size, dim) = cache.dims4()?;
                let starts = (0..heads).map(|h| (h * size) as u32).collect();
                (heads, dim, dim, heads * size, starts)
            }
        })
    }

    /// Rows `[heads, n, dim]` of the tokens in blocks `block`, at offsets `offset`: one
    /// `index_select` over the cache viewed as rows of its longest contiguous run.
    fn rows(self, cache: &Tensor, block: &Tensor, offset: &Tensor) -> Result<Tensor> {
        let (heads, dim, run, per_block, starts) = self.runs(cache)?;
        let runs = cache.elem_count() / run;
        if runs > u32::MAX as usize {
            hanzo_ml::bail!("QSA: a paged cache of {runs} rows is past u32 indexing");
        }
        let (n, dev) = (block.dim(0)?, cache.device());
        let m = starts.len();
        let base = block
            .broadcast_mul(&Tensor::new(&[per_block as u32], dev)?)?
            .add(offset)?;
        let idx = base
            .unsqueeze(1)?
            .broadcast_add(&Tensor::from_vec(starts, (1, m), dev)?)?
            .flatten_all()?;
        cache
            .reshape((runs, run))?
            .index_select(&idx, 0)?
            .reshape((n, heads, dim))?
            .transpose(0, 1)?
            .contiguous()
    }
}

/// Where tokens of one sequence sit in a cache entry, on the entry's device.
#[derive(Clone)]
pub(crate) enum At {
    /// Positions on an eager cache's sequence axis, `[n]` u32.
    Eager(Tensor),
    /// Physical blocks and offsets in them, `[n]` u32 each.
    Paged { block: Tensor, offset: Tensor },
}

impl At {
    /// `tokens` of a sequence whose paged block table is `table` (blocks of `size` tokens), or of
    /// an eager sequence without one.
    fn new(tokens: &[u32], table: Option<(&[u32], usize)>, dev: &Device) -> Result<Self> {
        let n = tokens.len();
        let Some((table, size)) = table else {
            return Ok(Self::Eager(Tensor::from_slice(tokens, n, dev)?));
        };
        let (mut block, mut offset) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for &t in tokens {
            let t = t as usize;
            let Some(&b) = table.get(t / size) else {
                hanzo_ml::bail!("QSA: token {t} lies past a {}-block table", table.len());
            };
            block.push(b);
            offset.push((t % size) as u32);
        }
        Ok(Self::Paged {
            block: Tensor::from_vec(block, n, dev)?,
            offset: Tensor::from_vec(offset, n, dev)?,
        })
    }

    fn to_device(&self, dev: &Device) -> Result<Self> {
        Ok(match self {
            Self::Eager(t) => Self::Eager(t.to_device(dev)?),
            Self::Paged { block, offset } => Self::Paged {
                block: block.to_device(dev)?,
                offset: offset.to_device(dev)?,
            },
        })
    }
}

/// One sequence's `(K, V)` pair of a cache entry, read by token.
enum Pair<'a> {
    /// Eager tensors `[1, heads, capacity, dim]`, contiguous.
    Eager(Tensor, Tensor),
    /// A paged pair and the sequence's block table.
    Paged {
        k: &'a Tensor,
        v: &'a Tensor,
        layout: (Layout, Layout),
        table: &'a [u32],
        size: usize,
    },
}

impl<'a> Pair<'a> {
    /// Sequence `b`'s rows of an eager cache.
    fn eager(cache: &KvCache, b: usize) -> Result<Self> {
        let KvCache::Normal { k, v } = cache else {
            hanzo_ml::bail!("QSA reads a normal eager KV cache");
        };
        let (Some(k), Some(v)) = (k.all_data(), v.all_data()) else {
            hanzo_ml::bail!("QSA read an empty eager KV cache");
        };
        Ok(Self::Eager(k.narrow(0, b, 1)?, v.narrow(0, b, 1)?))
    }

    /// A paged pair and the block table of the sequence reading it.
    fn paged(cache: &'a (Tensor, Tensor), table: &'a [u32], size: usize) -> Result<Self> {
        let (k, v) = cache;
        Ok(Self::Paged {
            k,
            v,
            layout: Layout::of(k, v)?,
            table,
            size,
        })
    }

    fn at(&self, tokens: &[u32]) -> Result<At> {
        match self {
            Self::Eager(k, _) => At::new(tokens, None, k.device()),
            Self::Paged { k, table, size, .. } => At::new(tokens, Some((table, *size)), k.device()),
        }
    }

    /// K rows `[heads, n, dim]` of the tokens at `at`.
    fn keys(&self, at: &At) -> Result<Tensor> {
        self.rows(at, true)
    }

    /// V rows `[heads, n, dim]` of the tokens at `at`.
    fn values(&self, at: &At) -> Result<Tensor> {
        self.rows(at, false)
    }

    fn rows(&self, at: &At, key: bool) -> Result<Tensor> {
        match (self, at) {
            (Self::Eager(k, v), At::Eager(ids)) => {
                let cache = if key { k } else { v };
                cache.index_select(ids, 2)?.squeeze(0)
            }
            (Self::Paged { k, v, layout, .. }, At::Paged { block, offset }) => {
                let (cache, layout) = if key { (k, layout.0) } else { (v, layout.1) };
                layout.rows(cache, block, offset)
            }
            _ => hanzo_ml::bail!("QSA: tokens placed for one kind of cache, read from another"),
        }
    }
}

/// Merge of two ascending token lists.
fn union(a: &[u32], b: &[u32]) -> Vec<u32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                out.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                out.push(a[i]);
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}

/// Attention of one sequence's queries `q` `[heads, n, head_dim]` over their selections `sel`,
/// as `[heads, n, head_dim]`: the QSA kernel's softmax over exactly the selected tokens, by
/// [`softmax_pv`].
///
/// Consecutive queries attend together while their union stays within [`UNION`] tokens and
/// [`LOGITS`] mask entries, and while its mask keeps at least half of what it scores. Each group
/// gathers the union's rows once, with the `heads / kv_heads` query heads sharing a K/V head
/// folded into its rows, so no K/V row is repeated.
fn sparse(q: &Tensor, sel: &[Vec<u32>], main: &Pair, kv_heads: usize) -> Result<Tensor> {
    let (heads, n, head_dim) = q.dims3()?;
    let fold = heads / kv_heads;
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let (mut tokens, mut kept, mut j) = (sel[i].clone(), sel[i].len(), i + 1);
        while j < n {
            let merged = union(&tokens, &sel[j]);
            let rows = j + 1 - i;
            let area = rows * merged.len();
            if merged.len() > UNION || area > LOGITS || area > 2 * (kept + sel[j].len()) {
                break;
            }
            (tokens, kept, j) = (merged, kept + sel[j].len(), j + 1);
        }
        let rows = j - i;
        let width = tokens.len();
        let mut mask = vec![0u8; rows * width];
        for (row, s) in mask.chunks_exact_mut(width).zip(&sel[i..j]) {
            let mut c = 0;
            for &t in s {
                while tokens[c] != t {
                    c += 1;
                }
                row[c] = 1;
            }
        }
        // A dropped token scores -1e30 below the rest: exp2 takes it to exactly 0, as -inf would.
        let mask = Tensor::from_vec(mask, (1, rows, width), q.device())?
            .to_dtype(DType::F32)?
            .affine(1e30, -1e30)?
            .repeat((fold, 1, 1))?
            .reshape((fold * rows, width))?;
        let at = main.at(&tokens)?;
        let (k, v) = (main.keys(&at)?, main.values(&at)?);
        let qg = q
            .narrow(1, i, rows)?
            .contiguous()?
            .reshape((kv_heads, fold * rows, head_dim))?;
        let y = softmax_pv(&qg, &k, &v, &mask)?;
        out.push(y.reshape((heads, rows, head_dim))?);
        i = j;
    }
    Tensor::cat(&out, 1)
}

/// One forward's sequences, as every QSA layer reads them.
pub(crate) struct Batch {
    cfg: Config,
    /// Each sequence's first position in this forward and its real tokens; the rest of its row is
    /// right padding.
    spans: Vec<(usize, usize)>,
    /// Paged: each sequence's block table, and the block size.
    tables: Option<(Vec<Vec<u32>>, usize)>,
    /// Where each sequence's group-end tokens sit, for reading its compressed keys; present when
    /// some query of the forward sees more than `blocks` groups.
    ends: Option<Vec<At>>,
}

impl Batch {
    /// `offsets[b]` is sequence `b`'s first position; `seq` the row length. With `paged` the real
    /// tokens are the slots its mapping does not pad and the tables are its block tables, else
    /// every row is real.
    pub(crate) fn new(
        cfg: Config,
        offsets: &[usize],
        seq: usize,
        paged: Option<&PagedAttentionInputMetadata>,
        dev: &Device,
    ) -> Result<Self> {
        let b_sz = offsets.len();
        let pick = |map: &std::collections::HashMap<hanzo_ml::DeviceLocation, Tensor>| {
            map.get(&dev.location())
                .or_else(|| map.values().next())
                .cloned()
        };
        let (lens, tables) = match paged {
            None => (vec![seq; b_sz], None),
            Some(meta) => {
                let slots = pick(&meta.slot_mappings)
                    .ok_or_else(|| hanzo_ml::Error::msg("QSA: no slot mapping"))?
                    .to_device(&Device::Cpu)?
                    .flatten_all()?
                    .to_vec1::<i64>()?;
                if slots.len() != b_sz * seq {
                    hanzo_ml::bail!("QSA: {} slots for {b_sz} rows of {seq}", slots.len());
                }
                let lens = slots
                    .chunks_exact(seq)
                    .map(|row| row.iter().filter(|&&s| s >= 0).count())
                    .collect();
                let tables: Option<(Vec<Vec<u32>>, usize)> =
                    match (meta.block_tables.as_ref().and_then(pick), meta.block_size) {
                        (Some(tables), Some(size)) => {
                            let tables = tables.to_device(&Device::Cpu)?.to_vec2::<u32>()?;
                            // A decode forward repeats each sequence's table once per query row.
                            let per = (tables.len() / b_sz.max(1)).max(1);
                            Some((tables.into_iter().step_by(per).collect(), size))
                        }
                        _ => None,
                    };
                (lens, tables)
            }
        };
        let spans: Vec<(usize, usize)> = offsets.iter().copied().zip(lens).collect();
        // A query at `p` drops a group once it sees more than `blocks`: from `p = width` on.
        let sparse = spans.iter().any(|&(a, n)| a + n > cfg.width());
        let ends = if sparse {
            let ends = spans
                .iter()
                .enumerate()
                .map(|(b, &(a, n))| {
                    let ends: Vec<u32> = (1..=(a + n) / cfg.ratio)
                        .map(|j| (j * cfg.ratio - 1) as u32)
                        .collect();
                    match (paged, &tables) {
                        (None, _) => At::new(&ends, None, dev),
                        (Some(_), Some((t, size))) => At::new(&ends, Some((&t[b], *size)), dev),
                        (Some(_), None) => {
                            hanzo_ml::bail!("QSA: a paged forward without block tables")
                        }
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            Some(ends)
        } else {
            None
        };
        Ok(Self {
            cfg,
            spans,
            tables,
            ends,
        })
    }

    /// Sequence `b`'s pair of a cache entry: paged when the forward is, else from `eager`.
    fn pair<'a>(
        &'a self,
        b: usize,
        eager: &KvCache,
        paged: Option<&'a (Tensor, Tensor)>,
    ) -> Result<Pair<'a>> {
        match (paged, &self.tables) {
            (Some(cache), Some((tables, size))) => Pair::paged(cache, &tables[b], *size),
            (None, _) => Pair::eager(eager, b),
            (Some(_), None) => hanzo_ml::bail!("QSA: a paged cache without block tables"),
        }
    }
}

/// A layer's paged K/V entry, its paged index entry, and the forward's metadata.
pub(crate) type Paged<'a> = (
    &'a (Tensor, Tensor),
    &'a (Tensor, Tensor),
    &'a PagedAttentionInputMetadata,
);

/// One attention layer's caches in a forward.
pub(crate) struct Store<'a> {
    /// The eager K/V, which the paged path leaves alone.
    pub kv: &'a mut KvCache,
    /// The eager index.
    pub index: &'a mut KvCache,
    pub paged: Option<Paged<'a>>,
}

/// One QSA attention layer on `u` `[B, S, hidden]`: `attn`'s projections and output around
/// attention restricted to `indexer`'s selections, as `attn.forward` returns it.
pub(crate) fn forward(
    attn: &GatedFullAttention,
    indexer: &Indexer,
    u: &Tensor,
    mask: &AttentionMask,
    cos_sin: &(Tensor, Tensor),
    batch: &Batch,
    store: Store<'_>,
) -> Result<Tensor> {
    let (b_sz, seq, _) = u.dims3()?;
    let cfg = indexer.cfg;
    if cfg != batch.cfg || batch.spans.len() != b_sz {
        hanzo_ml::bail!("QSA: a forward of {b_sz} rows read with another batch");
    }
    let (q, k, v, gate) = attn.project(u, cos_sin)?;
    let (iq, raw) = indexer.project(u, cos_sin)?;
    let (main_paged, index_paged, meta) = match store.paged {
        Some((main, index, meta)) => (Some(main), Some(index), Some(meta)),
        None => (None, None, None),
    };

    // The index rows of every sequence's chunk, written before anything reads them.
    let mut closed = Vec::with_capacity(b_sz);
    for (b, &(a, n)) in batch.spans.iter().enumerate() {
        let p = a % cfg.ratio;
        let prior = if p == 0 || p + n < cfg.ratio {
            None
        } else {
            let index = batch.pair(b, store.index, index_paged)?;
            let tokens: Vec<u32> = (a - p..a).map(|t| t as u32).collect();
            Some(index.values(&index.at(&tokens)?)?.squeeze(0)?)
        };
        let raw = raw.get(b)?.narrow(0, 0, n)?;
        closed.push(indexer.closed(attn, &raw, prior.as_ref(), a, seq)?);
    }
    let closed = Tensor::stack(&closed, 0)?;
    match (index_paged, meta) {
        (Some(index), Some(meta)) => {
            let rows = |t: &Tensor| {
                t.reshape((b_sz * seq, 1, cfg.dim))?
                    .to_dtype(index.0.dtype())
            };
            attn.write_paged(&rows(&closed)?, &rows(&raw)?, index, meta)?
        }
        _ => {
            store
                .index
                .append(&closed.unsqueeze(1)?, &raw.unsqueeze(1)?)?;
        }
    }

    let Some(ends) = batch.ends.as_ref() else {
        let paged = main_paged.zip(meta).map(|(c, m)| (c.clone(), m));
        let y = attn.attend(&q, &k, &v, mask, store.kv, paged)?;
        return attn.output(&y, &gate, u.dtype());
    };

    let (heads, kv_heads, head_dim) = attn.heads();
    match (main_paged, meta) {
        (Some(main), Some(meta)) => {
            // `[B, heads, S, dim]` as the slot mapping's `[B·S, heads, dim]` rows.
            let rows = |t: &Tensor| t.transpose(1, 2)?.reshape((b_sz * seq, kv_heads, head_dim));
            attn.write_paged(&rows(&k)?, &rows(&v)?, main, meta)?
        }
        _ => {
            store.kv.append(&k, &v)?;
        }
    }
    let mut ys = Vec::with_capacity(b_sz);
    for (b, &(a, n)) in batch.spans.iter().enumerate() {
        let kc = if (a + n) / cfg.ratio == 0 {
            Tensor::zeros((0, cfg.dim), raw.dtype(), raw.device())?
        } else {
            let index = batch.pair(b, store.index, index_paged)?;
            index.keys(&ends[b].to_device(q.device())?)?.squeeze(0)?
        };
        let pos: Vec<usize> = (a..a + n).collect();
        let sel = select(&iq.get(b)?.narrow(0, 0, n)?, &kc, &pos, &cfg)?;
        let main = batch.pair(b, store.kv, main_paged)?;
        let y = sparse(&q.get(b)?.narrow(1, 0, n)?, &sel, &main, kv_heads)?;
        let y = Tensor::cat(
            &[
                &y,
                &Tensor::zeros((heads, seq - n, head_dim), y.dtype(), y.device())?,
            ],
            1,
        )?;
        ys.push(y.transpose(0, 1)?.reshape((seq, heads * head_dim))?);
    }
    attn.output(&Tensor::stack(&ys, 0)?, &gate, u.dtype())
}

#[cfg(test)]
mod reference {
    //! Plain f64 transcriptions of each rule.

    /// Gemma RMS norm with its `1 + w` already in `w`.
    pub(super) fn norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
        x.iter().zip(w).map(|(v, w)| v * r * w).collect()
    }

    /// NeoX rotate-half over the leading `rot` dims at `pos`, frequency `i` being
    /// `theta^(-2i/rot)`.
    pub(super) fn rope(x: &mut [f64], pos: usize, rot: usize, theta: f64) {
        let half = rot / 2;
        for i in 0..half {
            let a = pos as f64 * theta.powf(-2.0 * i as f64 / rot as f64);
            let (x0, x1) = (x[i], x[i + half]);
            x[i] = x0 * a.cos() - x1 * a.sin();
            x[i + half] = x1 * a.cos() + x0 * a.sin();
        }
    }

    pub(super) fn linear(x: &[f64], w: &[f64]) -> Vec<f64> {
        w.chunks(x.len())
            .map(|r| r.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Normed, roped queries `[heads·dim]` and the raw key of one token.
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

    /// The group mean, rounded to bf16.
    pub(super) fn pool(raw: &[Vec<f64>]) -> Vec<f64> {
        (0..raw[0].len())
            .map(|d| {
                let mean = raw.iter().map(|k| k[d]).sum::<f64>() / raw.len() as f64;
                half::bf16::from_f64(mean).to_f64()
            })
            .collect()
    }

    /// Group scores of a query `[heads·dim]` at `p` over its visible compressed keys.
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
    /// groups outrank it, outranking meaning a higher score or an equal one at a lower index;
    /// kept groups expand to their tokens, then the tail.
    pub(super) fn select(
        q: &[f64],
        kc: &[Vec<f64>],
        p: usize,
        ratio: usize,
        blocks: usize,
    ) -> Vec<u32> {
        let s = scores(q, kc, p, ratio);
        let mut row = Vec::new();
        for j in 0..s.len() {
            let above = (0..s.len())
                .filter(|&i| s[i] > s[j] || (s[i] == s[j] && i < j))
                .count();
            if above < blocks {
                row.extend((j * ratio..(j + 1) * ratio).map(|t| t as u32));
            }
        }
        row.extend(((p + 1) / ratio * ratio..=p).map(|t| t as u32));
        row
    }

    /// Sparse GQA of one query `[heads·dim]` over the rows `tokens` of `k` and `v`
    /// `[token, kv_heads, dim]`: softmax of `scale·q·k` over them.
    pub(super) fn attend(
        q: &[f64],
        k: &[f64],
        v: &[f64],
        tokens: &[u32],
        kv_heads: usize,
        dim: usize,
        scale: f64,
    ) -> Vec<f64> {
        let heads = q.len() / dim;
        let mut out = vec![0.0; heads * dim];
        for (h, (qh, oh)) in q.chunks(dim).zip(out.chunks_mut(dim)).enumerate() {
            let at = |t: u32| (t as usize * kv_heads + h / (heads / kv_heads)) * dim;
            let logit: Vec<f64> = tokens
                .iter()
                .map(|&t| scale * qh.iter().zip(&k[at(t)..]).map(|(a, b)| a * b).sum::<f64>())
                .collect();
            let max = logit.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = logit.iter().map(|l| (l - max).exp()).collect();
            let z: f64 = w.iter().sum();
            for (&t, w) in tokens.iter().zip(&w) {
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

    /// Max error within `tol` of the reference's largest magnitude (at least 1).
    fn within(got: &Tensor, want: &[f64], tol: f64, what: &str) -> Result<()> {
        let got = f64s(got)?;
        assert_eq!(got.len(), want.len(), "{what}: length");
        let scale = want.iter().fold(1f64, |m, v| m.max(v.abs()));
        let err = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(err <= tol * scale, "{what}: error {err} at scale {scale}");
        Ok(())
    }

    fn close(got: &Tensor, want: &[f64], what: &str) -> Result<()> {
        within(got, want, 1e-5, what)
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
    /// BF16 as the real file stores them.
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
        let arch = gguf_file::Value::String("zen6".to_string());
        let (heads, dim, top_k) = (u32v(cfg.heads), u32v(cfg.dim), u32v(cfg.blocks * cfg.ratio));
        let metadata = [
            ("general.architecture", &arch),
            ("zen6.attention.indexer.head_count", &heads),
            ("zen6.attention.indexer.key_length", &dim),
            ("zen6.attention.indexer.top_k", &top_k),
            ("zen6.attention.compress_ratios", &ratios),
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
                path_prefix: "zen6",
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
        .map(|(k, v)| (format!("zen6.attention.{k}"), v))
        .collect();
        let md = ContentMetadata {
            path_prefix: "zen6",
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
    /// rounds once, before RoPE (the query and the compressed key alike). RoPE at position 0 is
    /// the identity, so each output is that one rounding of the f64 norm of the bf16 values the
    /// indexer normed. Rounding `1 + w` to bf16 first moves many outputs an ulp. The bound is half
    /// a bf16 ulp, not 1e-5, because the bf16 rounding is what is under test.
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
    /// cut the ranking inside a run of equal scores, zeros and non-zeros both. The queries arrive
    /// in two scrambled orders so the scored rows are not contiguous.
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
        let keys: Vec<Vec<f64>> = kc.chunks(d).map(wide).collect();
        let kc = Tensor::from_vec(kc, (groups, d), &dev)?;
        for order in [
            (0..len).collect::<Vec<usize>>(),
            (0..len).map(|i| (i * 17) % len).collect(),
        ] {
            let rows: Vec<f32> = order
                .iter()
                .flat_map(|&p| q[p * cfg.heads * d..(p + 1) * cfg.heads * d].to_vec())
                .collect();
            let sel = select(
                &Tensor::from_vec(rows, (len, cfg.heads, d), &dev)?,
                &kc,
                &order,
                &cfg,
            )?;
            let (mut ties, mut zeros) = (0, 0);
            for (row, &p) in sel.iter().zip(&order) {
                let x = wide(&q[p * cfg.heads * d..(p + 1) * cfg.heads * d]);
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
        }
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
            )?;
            for (p, row) in sel.iter().enumerate() {
                if p < width {
                    assert_eq!(*row, (0..=p as u32).collect::<Vec<_>>(), "len {len}, p {p}");
                } else {
                    assert_eq!(row.len(), cfg.blocks * cfg.ratio, "len {len}, query at {p}");
                }
            }
        }
        Ok(())
    }

    /// Paged caches holding rows `[token, heads, dim]` of tokens placed through `table`, NaN in
    /// every other slot: the standard `[blocks, heads, dim / x, size, x]` and
    /// `[blocks, heads, dim, size]` by the index arithmetic of the engine's `reshape_and_cache`
    /// kernels, and FlashInfer's `[blocks, heads, size, dim]` of each.
    #[allow(clippy::type_complexity)]
    fn paged(
        k: &[f32],
        v: &[f32],
        table: &[u32],
        (blocks, heads, dim, size, x): (usize, usize, usize, usize, usize),
    ) -> Result<((Tensor, Tensor), (Tensor, Tensor))> {
        let n = blocks * heads * dim * size;
        let (mut kc, mut vc) = (vec![f32::NAN; n], vec![f32::NAN; n]);
        let (mut kh, mut vh) = (vec![f32::NAN; n], vec![f32::NAN; n]);
        for t in 0..k.len() / (heads * dim) {
            let (b, o) = (table[t / size] as usize, t % size);
            for h in 0..heads {
                for d in 0..dim {
                    let src = (t * heads + h) * dim + d;
                    kc[(((b * heads + h) * (dim / x) + d / x) * size + o) * x + d % x] = k[src];
                    vc[((b * heads + h) * dim + d) * size + o] = v[src];
                    kh[((b * heads + h) * size + o) * dim + d] = k[src];
                    vh[((b * heads + h) * size + o) * dim + d] = v[src];
                }
            }
        }
        let dev = Device::Cpu;
        Ok((
            (
                Tensor::from_vec(kc, (blocks, heads, dim / x, size, x), &dev)?,
                Tensor::from_vec(vc, (blocks, heads, dim, size), &dev)?,
            ),
            (
                Tensor::from_vec(kh, (blocks, heads, size, dim), &dev)?,
                Tensor::from_vec(vh, (blocks, heads, size, dim), &dev)?,
            ),
        ))
    }

    /// Rows read through a block table from both paged layouts, and from an eager cache, are the
    /// rows written; unwritten slots, which hold NaN, are never read.
    #[test]
    fn rows_read_back_through_every_layout() -> Result<()> {
        let dev = Device::Cpu;
        let (heads, dim, size, x, len) = (2usize, 8usize, 4usize, 4usize, 13usize);
        let table = [5u32, 1, 3, 0];
        let mut rng = StdRng::seed_from_u64(0x7173_6106);
        let k = grid(&mut rng, len * heads * dim, 64);
        let v = grid(&mut rng, len * heads * dim, 64);
        let tokens = [12u32, 0, 7, 3, 4, 11];
        let want = |x: &[f32]| -> Vec<f64> {
            (0..heads)
                .flat_map(|h| {
                    tokens.iter().flat_map(move |&t| {
                        x[(t as usize * heads + h) * dim..(t as usize * heads + h + 1) * dim]
                            .to_vec()
                    })
                })
                .map(f64::from)
                .collect()
        };

        let (standard, hnd) = paged(&k, &v, &table, (6, heads, dim, size, x))?;
        let pair = Pair::paged(&standard, &table, size)?;
        let at = pair.at(&tokens)?;
        close(&pair.keys(&at)?, &want(&k), "standard keys")?;
        close(&pair.values(&at)?, &want(&v), "standard values")?;

        let pair = Pair::Paged {
            k: &hnd.0,
            v: &hnd.1,
            layout: (Layout::Hnd, Layout::Hnd),
            table: &table,
            size,
        };
        let at = pair.at(&tokens)?;
        close(&pair.keys(&at)?, &want(&k), "flashinfer keys")?;
        close(&pair.values(&at)?, &want(&v), "flashinfer values")?;

        // An eager cache `[1, heads, capacity, dim]` with room past the sequence.
        let eager = |x: &[f32]| -> Result<Tensor> {
            Tensor::from_vec(x.to_vec(), (1, len, heads, dim), &dev)?
                .transpose(1, 2)?
                .pad_with_zeros(2, 0, 3)?
                .contiguous()
        };
        let pair = Pair::Eager(eager(&k)?, eager(&v)?);
        let at = pair.at(&tokens)?;
        close(&pair.keys(&at)?, &want(&k), "eager keys")?;
        close(&pair.values(&at)?, &want(&v), "eager values")
    }

    /// [`attend`] over selections read from a paged cache equals the f64 reference restricted to
    /// each query's tokens, whether the queries group or not, and differs from plain causal
    /// attention. Selections of neighbouring queries overlap by construction, so some groups hold
    /// several queries.
    #[test]
    fn attend_equals_reference_over_selections() -> Result<()> {
        let dev = Device::Cpu;
        let (heads, kv_heads, hd, len, size) = (4usize, 2usize, 8usize, 40usize, 8usize);
        let table = [6u32, 2, 9, 0, 4];
        let mut rng = StdRng::seed_from_u64(0x7173_6104);
        let mut uniform =
            |n: usize| -> Vec<f32> { (0..n).map(|_| rng.random_range(-1.0f32..1.0)).collect() };
        let pos: Vec<usize> = (24..40).collect();
        let (q, k, v) = (
            uniform(pos.len() * heads * hd),
            uniform(len * kv_heads * hd),
            uniform(len * kv_heads * hd),
        );
        // Each query keeps groups 0 and 2 and one that walks with it, all behind it, then its
        // tail.
        let sel: Vec<Vec<u32>> = pos
            .iter()
            .map(|&p| {
                [0, 2, p / 4 - 3]
                    .into_iter()
                    .flat_map(|j| j * 4..j * 4 + 4)
                    .chain((p + 1) / 4 * 4..=p)
                    .map(|t| t as u32)
                    .collect()
            })
            .collect();
        let (cache, _) = paged(&k, &v, &table, (10, kv_heads, hd, size, 4))?;
        let pair = Pair::paged(&cache, &table, size)?;
        let scale = 1.0 / (hd as f64).sqrt();
        // `[heads, n, hd]` from rows `[n, heads, hd]`.
        let qt = Tensor::from_vec(q.clone(), (pos.len(), heads, hd), &dev)?
            .transpose(0, 1)?
            .contiguous()?;
        let got = sparse(&qt, &sel, &pair, kv_heads)?;
        let want: Vec<f64> = q
            .chunks(heads * hd)
            .zip(&sel)
            .flat_map(|(x, s)| {
                reference::attend(&wide(x), &wide(&k), &wide(&v), s, kv_heads, hd, scale)
            })
            .collect();
        within(&got.transpose(0, 1)?, &want, 1e-5, "sparse attention")?;

        let causal: Vec<Vec<u32>> = pos.iter().map(|&p| (0..=p as u32).collect()).collect();
        let dense = sparse(&qt, &causal, &pair, kv_heads)?;
        let gap = (dense - &got)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(gap > 1e-2, "the selection changed nothing ({gap})");
        Ok(())
    }

    /// The layer shape: 4 query heads over 2 K/V heads of 16, 8 of them rotary, and an indexer of
    /// 2 heads of 16 keeping 2 groups of 4, a width of 11 tokens.
    const LAYER: (usize, usize, usize) = (4, 2, 16);
    const LAYER_QSA: Config = Config {
        heads: 2,
        dim: 16,
        ratio: 4,
        blocks: 2,
    };

    /// One attention block `blk.3` with its indexer, eager, and the rotary it shares with it.
    fn layer(
        dir: &std::path::Path,
    ) -> Result<(GatedFullAttention, Indexer, Qwen3VLRotaryEmbedding)> {
        use crate::models::quantized_qwen3_5_moe::PropsGGUF;
        let (heads, kv_heads, hd) = LAYER;
        let path = dir.join("layer.gguf");
        let mut rng = StdRng::seed_from_u64(0x7173_6107);
        fixture(&path, &LAYER_QSA, &mut rng)?;
        let mut tensors = Vec::new();
        for (name, shape) in [
            ("attn_q", vec![2 * heads * hd, HIDDEN]),
            ("attn_k", vec![kv_heads * hd, HIDDEN]),
            ("attn_v", vec![kv_heads * hd, HIDDEN]),
            ("attn_output", vec![HIDDEN, heads * hd]),
            ("attn_q_norm", vec![hd]),
            ("attn_k_norm", vec![hd]),
        ] {
            let n: usize = shape.iter().product();
            let shift = if shape.len() == 1 { 1.0 } else { 0.0 };
            let data: Vec<f32> = (0..n)
                .map(|_| shift + rng.random_range(-0.4f32..0.4))
                .collect();
            let t = QTensor::quantize(
                &Tensor::from_vec(data, shape, &Device::Cpu)?,
                GgmlDType::F32,
            )?;
            tensors.push((format!("blk.3.{name}.weight"), t));
        }
        // The indexer's file, with the attention tensors added.
        let mut files = [std::fs::File::open(&path).map_err(hanzo_ml::Error::msg)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        for name in ["q_proj", "k_proj", "q_norm", "k_norm"] {
            let name = format!("blk.3.indexer.{name}.weight");
            let t = ct.tensor(&name, &Device::Cpu)?;
            tensors.push((name, t));
        }
        let metadata: Vec<(String, gguf_file::Value)> = ct
            .get_metadata()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let full = dir.join("full.gguf");
        let mut file = std::fs::File::create(&full).map_err(hanzo_ml::Error::msg)?;
        gguf_file::write(
            &mut file,
            &metadata
                .iter()
                .map(|(k, v)| (k.as_str(), v))
                .collect::<Vec<_>>(),
            &tensors
                .iter()
                .map(|(n, t)| (n.as_str(), t))
                .collect::<Vec<_>>(),
        )?;
        let mut files = [std::fs::File::open(&full).map_err(hanzo_ml::Error::msg)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let props = PropsGGUF {
            head_count: heads,
            head_count_kv: kv_heads,
            block_count: 4,
            embedding_length: HIDDEN,
            rms_norm_eps: EPS,
            max_seq_len: 64,
            rope_freq_base: THETA,
            head_dim: hd,
            rot_dim: ROT,
            mrope_section: vec![2, 1, 1],
            full_attention_interval: 4,
            conv_kernel: 4,
            head_k_dim: 4,
            head_v_dim: 4,
            num_k_heads: 1,
            num_v_heads: 1,
            num_experts: None,
            num_experts_per_tok: 0,
            moe_intermediate_size: 0,
            is_moe: false,
            nextn_predict_layers: 0,
        };
        let dev = Device::Cpu;
        let rotary = Qwen3VLRotaryEmbedding::new(THETA, ROT, &dev, vec![2, 1, 1])?;
        let attn = GatedFullAttention::load(
            &mut ct,
            "blk.3",
            &props,
            Arc::new(rotary.clone()),
            None,
            &dev,
            DType::F32,
        )?
        .qsa();
        let indexer = Indexer::from_gguf(&mut ct, "blk.3", LAYER_QSA, EPS, &dev)?;
        Ok((attn, indexer, rotary))
    }

    /// `u` through the layer in chunks of `chunks` tokens on fresh eager caches, the way the
    /// model feeds it: its causal mask for a prompt chunk, none for a single token.
    fn run(
        (attn, indexer, rotary): &(GatedFullAttention, Indexer, Qwen3VLRotaryEmbedding),
        u: &Tensor,
        chunks: &[usize],
        qsa: bool,
    ) -> Result<Tensor> {
        let dev = Device::Cpu;
        let (mut kv, mut index) = (
            KvCache::new_normal(2, 4096, 16),
            KvCache::new_normal(2, 4096, 16),
        );
        let (mut a, mut out) = (0, Vec::new());
        for &n in chunks {
            let pos: Vec<usize> = (a..a + n).collect();
            let cos_sin = rotary.compute_cos_sin(&planes(&pos, &dev)?, DType::F32)?;
            let mask = if n == 1 {
                AttentionMask::None
            } else {
                let m: Vec<f32> = (0..n)
                    .flat_map(|i| {
                        (0..a + n).map(move |t| if t <= a + i { 0.0 } else { f32::NEG_INFINITY })
                    })
                    .collect();
                AttentionMask::Custom(Tensor::from_vec(m, (n, a + n), &dev)?)
            };
            let x = u.narrow(1, a, n)?;
            out.push(if qsa {
                let batch = Batch::new(indexer.cfg, &[a], n, None, &dev)?;
                let store = Store {
                    kv: &mut kv,
                    index: &mut index,
                    paged: None,
                };
                forward(attn, indexer, &x, &mask, &cos_sin, &batch, store)?
            } else {
                attn.forward(&x, &mask, &cos_sin, &mut kv, None)?
            });
            a += n;
        }
        Tensor::cat(&out, 1)
    }

    /// Up to the width the layer is its dense attention, bit for bit, however the tokens arrive:
    /// the indexer's projections and index writes leave the attention alone.
    #[test]
    fn layer_within_width_is_dense_exactly() -> Result<()> {
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let layer = layer(dir.path())?;
        let width = LAYER_QSA.width();
        let mut rng = StdRng::seed_from_u64(0x7173_6108);
        let u = Tensor::from_vec(
            grid(&mut rng, width * HIDDEN, 64),
            (1, width, HIDDEN),
            &Device::Cpu,
        )?;
        for chunks in [vec![width], vec![5, 1, 5], vec![3, 4, 1, 1, 2]] {
            let got = run(&layer, &u, &chunks, true)?;
            let want = run(&layer, &u, &chunks, false)?;
            let gap = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert_eq!(gap, 0.0, "chunks {chunks:?}");
        }
        Ok(())
    }

    /// Past the width, the layer equals attention restricted to each query's brute-force
    /// selection over the compressed keys of every group closed so far, however the tokens
    /// arrive: in one pass, or in chunks that split groups and cross the width, with single
    /// tokens between. The selection is the f64 reference's over the same projections, so the
    /// index's bookkeeping -- the open group's raw keys carried across chunks, each group closed
    /// once at its last token and roped at its first -- is what is under test, with the scores,
    /// the ranking and the attention.
    #[test]
    fn layer_equals_restricted_attention() -> Result<()> {
        let dev = Device::Cpu;
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let layer = layer(dir.path())?;
        let (attn, indexer, rotary) = &layer;
        let (heads, kv_heads, hd) = LAYER;
        let cfg = LAYER_QSA;
        let len = 37;
        let mut rng = StdRng::seed_from_u64(0x7173_6109);
        let u = Tensor::from_vec(grid(&mut rng, len * HIDDEN, 64), (1, len, HIDDEN), &dev)?;

        // The reference, from whole-sequence projections.
        let pos: Vec<usize> = (0..len).collect();
        let cos_sin = rotary.compute_cos_sin(&planes(&pos, &dev)?, DType::F32)?;
        let (q, k, v, gate) = attn.project(&u, &cos_sin)?;
        let (iq, raw) = indexer.project(&u, &cos_sin)?;
        let groups = len / cfg.ratio;
        let first: Vec<usize> = (0..groups).map(|j| j * cfg.ratio).collect();
        let kc = indexer.compress(
            &raw.get(0)?.narrow(0, 0, groups * cfg.ratio)?,
            &rotary.compute_cos_sin(&planes(&first, &dev)?, DType::F32)?,
        )?;
        let keys: Vec<Vec<f64>> = f64s(&kc)?.chunks(cfg.dim).map(<[f64]>::to_vec).collect();
        let iq = f64s(&iq)?;
        let rows = |t: &Tensor| f64s(&t.get(0)?.transpose(0, 1)?.contiguous()?);
        let (qr, kr, vr) = (rows(&q)?, rows(&k)?, rows(&v)?);
        let scale = 1.0 / (hd as f64).sqrt();
        let mut dropped = 0;
        let y: Vec<f32> = (0..len)
            .flat_map(|p| {
                let x = &iq[p * cfg.heads * cfg.dim..(p + 1) * cfg.heads * cfg.dim];
                let sel = reference::select(x, &keys, p, cfg.ratio, cfg.blocks);
                dropped += usize::from(sel.len() <= p);
                let qp = &qr[p * heads * hd..(p + 1) * heads * hd];
                reference::attend(qp, &kr, &vr, &sel, kv_heads, hd, scale)
            })
            .map(|x| x as f32)
            .collect();
        assert!(dropped > len / 2, "only {dropped} queries dropped a group");
        let want = attn.output(
            &Tensor::from_vec(y, (1, len, heads * hd), &dev)?,
            &gate,
            DType::F32,
        )?;
        let want = f64s(&want)?;

        for chunks in [vec![len], vec![13, 1, 1, 9, 6, 7], vec![2, 10, 1, 24]] {
            let got = run(&layer, &u, &chunks, true)?;
            within(&got, &want, 1e-5, &format!("chunks {chunks:?}"))?;
        }
        Ok(())
    }
}
