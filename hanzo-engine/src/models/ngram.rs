#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

//! Hashed n-gram memory of Qwen3.8-Flash-Next (GGUF arch `qwen4exp`, `per_layer_token_embd` +
//! `blk.1.ple_*`; vLLM `Qwen4ExpPLELayer`).
//!
//! Each token's n-grams (orders 2..=ngram_size, `heads_per_ngram` hash heads each) pick rows of one
//! large table; the rows, concatenated, are the embedding `e`. The block gates a value projection of
//! `e` per residual stream, runs it through a causal depthwise conv dilated by the n-gram order, and
//! returns the delta the model adds to its streams.
//!
//! Source: vLLM `models/qwen4_exp/nvidia/ple_layer.py` and `model_state.py`.

use std::io::{Read, Seek};
use std::sync::Arc;

use hanzo_ml::quantized::QTensor;
use hanzo_ml::{bail, DType, Device, IndexOp, Result, Tensor, D};
use hanzo_quant::QuantMethod;

use crate::gguf::Content;
use crate::kv_cache::RecurrentLayerConfig;
use crate::models::gdn::{sigmoid, GdnLayerCache};
use crate::models::quantized_qwen3_5_moe::gguf_qmm;
use crate::utils::gguf_metadata::ContentMetadata;

/// The n-gram hash: token ids to table rows, computed on the host in u64.
pub(crate) struct Hash {
    /// Segment separator, `ple.eos_token_id`. Not the tokenizer's EOS.
    eos: u32,
    heads: usize,
    /// One multiplier per n-gram position, `mult[i]` for the token `i` places back.
    mult: Vec<u64>,
    /// Per head, orders 2..=ngram_size in turn: the head's first row and its (prime) row count.
    offset: Vec<u64>,
    size: Vec<u64>,
}

impl Hash {
    /// The constants the converter stored under `ple.*`; nothing is derived here.
    pub(crate) fn from_gguf(md: &ContentMetadata) -> Result<Self> {
        let int = |k: &str| md.get_value::<u32>(k).map_err(hanzo_ml::Error::msg);
        let ints = |k: &str| md.get_value::<Vec<u64>>(k).map_err(hanzo_ml::Error::msg);
        let order = int("ple.ngram_size")? as usize;
        let hash = Self {
            eos: int("ple.eos_token_id")?,
            heads: int("ple.heads_per_ngram")? as usize,
            mult: ints("ple.layer_multipliers")?,
            offset: ints("ple.head_offsets")?,
            size: ints("ple.head_vocab_sizes")?,
        };
        // The head tables sit end to end (ple_layer.py:229-246) and every row fits a u32 id.
        let end = hash
            .offset
            .iter()
            .zip(&hash.size)
            .try_fold(0u64, |at, (&o, &s)| (o == at).then_some(at + s));
        if order < 2
            || hash.mult.len() != order
            || hash.size.len() != (order - 1) * hash.heads
            || hash.offset.len() != hash.size.len()
            || !end.is_some_and(|end| end <= u64::from(u32::MAX))
        {
            bail!(
                "n-gram hash metadata is inconsistent: order {order}, {} heads, {} multipliers, {} offsets, {} sizes",
                hash.heads,
                hash.mult.len(),
                hash.offset.len(),
                hash.size.len()
            );
        }
        Ok(hash)
    }

    /// The hash from a safetensors checkpoint's I64 buffers at `vb`
    /// (`...ple.ple_embedding`: `layer_multipliers`, `ngram_heads_vocab_sizes`,
    /// `ngram_heads_offsets`), with `eos` the config's `eos_token_id` and `heads` its
    /// `heads_per_ngram`.
    pub(crate) fn new(vb: &hanzo_quant::ShardedVarBuilder, eos: u32, heads: usize) -> Result<Self> {
        let host = vb.clone().set_device(Device::Cpu);
        let u64s = |name: &str| -> Result<Vec<u64>> {
            Ok(host
                .get_unchecked_dtype(name, DType::I64)?
                .to_vec1::<i64>()?
                .into_iter()
                .map(|v| v as u64)
                .collect())
        };
        Self::checked(
            Self {
                eos,
                heads,
                mult: u64s("layer_multipliers")?,
                offset: u64s("ngram_heads_offsets")?,
                size: u64s("ngram_heads_vocab_sizes")?,
            },
            None,
        )
    }

    /// `hash` if its tables sit end to end, every row fits a u32 id and the counts agree.
    fn checked(hash: Self, order: Option<usize>) -> Result<Self> {
        let order = order.unwrap_or(hash.mult.len());
        let end = hash
            .offset
            .iter()
            .zip(&hash.size)
            .try_fold(0u64, |at, (&o, &s)| (o == at).then_some(at + s));
        if order < 2
            || hash.mult.len() != order
            || hash.size.len() != (order - 1) * hash.heads
            || hash.offset.len() != hash.size.len()
            || !end.is_some_and(|end| end <= u64::from(u32::MAX))
        {
            bail!(
                "n-gram hash is inconsistent: order {order}, {} heads, {} multipliers, {} offsets, {} sizes",
                hash.heads,
                hash.mult.len(),
                hash.offset.len(),
                hash.size.len()
            );
        }
        Ok(hash)
    }

    /// The n-gram order, `ple.ngram_size`.
    fn order(&self) -> usize {
        self.mult.len()
    }

    /// The table rows of each token of `chunk`, `(order - 1) · heads` per token, orders 2..=order
    /// in turn. `prior` is the sequence before the chunk; only its last `order - 1` tokens matter,
    /// and missing ones read as EOS (model_state.py:65-89).
    pub(crate) fn rows(&self, prior: &[u32], chunk: &[u32]) -> Vec<u32> {
        let span = self.order() - 1;
        let mut seq = vec![self.eos; span.saturating_sub(prior.len())];
        seq.extend_from_slice(&prior[prior.len().saturating_sub(span)..]);
        seq.extend_from_slice(chunk);
        let mut rows = Vec::with_capacity(chunk.len() * self.size.len());
        for t in span..seq.len() {
            // mixed_n = XOR over i < n of x[t-i]·mult[i] (ple_layer.py:426-431). An EOS at t-i cuts
            // off every older token, which then reads as EOS (:337-369); the token at t is never cut.
            // No product overflows: each multiplier is below 2^63 / vocab (:219).
            let mut mixed = u64::from(seq[t]) * self.mult[0];
            let mut cut = false;
            for n in 2..=self.order() {
                let back = seq[t + 1 - n];
                mixed ^= u64::from(if cut { self.eos } else { back }) * self.mult[n - 1];
                cut |= back == self.eos;
                // row = mixed_n mod size + offset, per head (:433).
                for h in (n - 2) * self.heads..(n - 1) * self.heads {
                    rows.push((mixed % self.size[h] + self.offset[h]) as u32);
                }
            }
        }
        rows
    }
}

/// The FP8 n-gram table of a safetensors checkpoint, read from its file mapping row by row.
///
/// The checkpoint splits the table into `shard_{i}` tensors of `rows_per_shard` rows that sit end
/// to end in one file; row `r` is `width` bytes at `shards[r / rows_per_shard] + (r %
/// rows_per_shard) · width`. The table's data starts at an arbitrary (only 2-byte aligned) file
/// offset, so rows are copied as bytes. Every row is E4M3 over one global BF16 scale.
pub(crate) struct Fp8Table {
    map: Arc<memmap2::Mmap>,
    /// Absolute byte offset of each shard's data in the mapping.
    shards: Vec<usize>,
    rows_per_shard: usize,
    width: usize,
    /// `ngram_embedding.weight_scale`, BF16, as f32.
    scale: f32,
}

impl Fp8Table {
    /// The table under `prefix` (`...ngram_embedding`) in `weights`: headers parsed, the file
    /// holding `shard_0` mapped, and the table's byte range advised random access.
    pub(crate) fn open(weights: &[std::path::PathBuf], prefix: &str) -> Result<Self> {
        use std::io::Read;
        for path in weights {
            let mut f = std::fs::File::open(path).map_err(hanzo_ml::Error::wrap)?;
            let mut len = [0u8; 8];
            f.read_exact(&mut len).map_err(hanzo_ml::Error::wrap)?;
            let n = u64::from_le_bytes(len) as usize;
            let mut buf = vec![0u8; n];
            f.read_exact(&mut buf).map_err(hanzo_ml::Error::wrap)?;
            let header: std::collections::HashMap<String, serde_json::Value> =
                serde_json::from_slice(&buf).map_err(hanzo_ml::Error::wrap)?;
            let entry = |name: &str| -> Option<(String, Vec<usize>, usize, usize)> {
                let v = header.get(name)?;
                let off = v["data_offsets"].as_array()?;
                Some((
                    v["dtype"].as_str()?.to_string(),
                    v["shape"].as_array()?.iter().map(|d| d.as_u64().unwrap_or(0) as usize).collect(),
                    8 + n + off[0].as_u64()? as usize,
                    8 + n + off[1].as_u64()? as usize,
                ))
            };
            let Some((dtype, shape, _, _)) = entry(&format!("{prefix}.shard_0.weight")) else {
                continue;
            };
            if dtype != "F8_E4M3" || shape.len() != 2 {
                bail!("{prefix}.shard_0 is {dtype} {shape:?}, expected 2-D F8_E4M3");
            }
            let (rows_per_shard, width) = (shape[0], shape[1]);
            let mut shards = Vec::new();
            let mut end = 0usize;
            while let Some((dtype, shape, start, stop)) =
                entry(&format!("{prefix}.shard_{}.weight", shards.len()))
            {
                if dtype != "F8_E4M3" || shape != [rows_per_shard, width] {
                    bail!("{prefix}.shard_{} is {dtype} {shape:?}", shards.len());
                }
                shards.push(start);
                end = end.max(stop);
            }
            let Some((sdtype, _, sstart, _)) = entry(&format!("{prefix}.weight_scale")) else {
                bail!("{prefix}.weight_scale is missing beside the table");
            };
            let map = unsafe { memmap2::Mmap::map(&f).map_err(hanzo_ml::Error::wrap)? };
            let raw = &map[sstart..sstart + 4];
            let scale = match sdtype.as_str() {
                "BF16" => half::bf16::from_le_bytes([raw[0], raw[1]]).to_f32(),
                "F32" => f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                d => bail!("{prefix}.weight_scale is {d}"),
            };
            #[cfg(unix)]
            {
                let first = shards[0];
                let page = 4096;
                let lo = first / page * page;
                let hi = end;
                unsafe {
                    libc::madvise(
                        map.as_ptr().add(lo) as *mut libc::c_void,
                        hi - lo,
                        libc::MADV_RANDOM,
                    );
                }
            }
            return Ok(Self {
                map: Arc::new(map),
                shards,
                rows_per_shard,
                width,
                scale,
            });
        }
        bail!("no weight file holds {prefix}.shard_0.weight")
    }

    /// The raw E4M3 bytes of `rows`, `width` each.
    pub(crate) fn bytes(&self, rows: &[u32]) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(rows.len() * self.width);
        for &r in rows {
            let r = r as usize;
            let shard = r / self.rows_per_shard;
            let Some(&base) = self.shards.get(shard) else {
                bail!("n-gram row {r} is past the table");
            };
            let at = base + (r % self.rows_per_shard) * self.width;
            out.extend_from_slice(&self.map[at..at + self.width]);
        }
        Ok(out)
    }
}

/// The table rows the hash picks from.
pub(crate) enum Table {
    /// `per_layer_token_embd`, (rows, width), left on the CPU where the mmap serves it.
    Gguf(QTensor),
    Fp8(Fp8Table),
}

/// The n-gram block: the hash, its table, and the gate + dilated conv that turn the gathered rows
/// into a delta for every residual stream.
pub(crate) struct Ngram {
    hash: Hash,
    table: Table,
    /// Embedding to one key per stream, (streams · hidden).
    key: Arc<dyn QuantMethod>,
    /// Embedding to one value shared by the streams, (hidden).
    value: Arc<dyn QuantMethod>,
    /// Per-stream RMS weights, (streams · hidden), f32, Gemma's 1 + w folded in by the converter.
    norm_query: Tensor,
    norm_key: Tensor,
    norm_conv: Tensor,
    /// Depthwise conv taps, (streams · hidden, kernel), bf16.
    conv: Tensor,
    eps: f64,
}

impl Ngram {
    /// Load the block at `blk.{layer}`.
    pub(crate) fn load<R: Read + Seek>(
        ct: &mut Content<'_, R>,
        hash: Hash,
        layer: usize,
        eps: f64,
        dev: &Device,
    ) -> Result<Self> {
        let table = ct.tensor("per_layer_token_embd.weight", &Device::Cpu)?;
        let (rows, _) = table.shape().dims2()?;
        let reach = hash.offset.last().zip(hash.size.last()).map(|(o, s)| o + s);
        if reach.is_none_or(|reach| reach > rows as u64) {
            bail!("n-gram table has {rows} rows; the hash reaches {reach:?}");
        }
        let mut get = |name: &str| ct.tensor(&format!("blk.{layer}.ple_{name}.weight"), dev);
        Ok(Self {
            key: gguf_qmm(get("key")?)?,
            value: gguf_qmm(get("value")?)?,
            norm_query: get("norm_query")?.dequantize(dev)?,
            norm_key: get("norm_key")?.dequantize(dev)?,
            norm_conv: get("norm_conv")?.dequantize(dev)?,
            // The taps are bf16 parameters in vLLM (ple_layer.py:589-598, 1152).
            conv: get("conv1d")?.dequantize(dev)?.to_dtype(DType::BF16)?,
            hash,
            table: Table::Gguf(table),
            eps,
        })
    }

    /// The block at `vb` (`...ple`) of a safetensors checkpoint: `key_proj`/`value_proj` bf16
    /// GEMMs, the three Gemma norms as `1 + w` in f32, the depthwise conv `[c, 1, k]` as bf16
    /// taps `[c, k]`, and the table and hash given.
    pub(crate) fn new(
        vb: &hanzo_quant::ShardedVarBuilder,
        hash: Hash,
        table: Fp8Table,
        eps: f64,
    ) -> Result<Self> {
        let dev = vb.device().clone();
        let host = vb.clone().set_device(Device::Cpu);
        let w = |name: &str| host.get_unchecked_dtype(&format!("{name}.weight"), DType::F32);
        let key = w("key_proj")?;
        let value = w("value_proj")?;
        let lin = |name: &str, t: &Tensor| {
            hanzo_quant::linear_no_bias(t.dim(1)?, t.dim(0)?, &None, vb.pp(name))
        };
        let gemma = |name: &str| -> Result<Tensor> { (w(name)?.to_device(&dev)? + 1.0) };
        let conv = w("conv1d")?.squeeze(1)?.to_dtype(DType::BF16)?.to_device(&dev)?;
        Ok(Self {
            key: lin("key_proj", &key)?,
            value: lin("value_proj", &value)?,
            norm_query: gemma("norm_query")?,
            norm_key: gemma("norm_key")?,
            norm_conv: gemma("norm_conv")?,
            conv,
            hash,
            table: Table::Fp8(table),
            eps,
        })
    }

    /// The conv history a sequence carries: (channels, rows), rows = (kernel - 1) · dilation, and
    /// the dilation is the n-gram order (ple_layer.py:551-552).
    pub(crate) fn history(&self) -> Result<(usize, usize)> {
        let (channels, kernel) = self.conv.dims2()?;
        Ok((channels, (kernel - 1) * self.hash.order()))
    }

    /// The side pool that carries the history across forwards: conv only, in `dtype`. vLLM keeps
    /// it in the model dtype (ple_layer.py:638-648).
    pub(crate) fn pool(&self, dtype: DType) -> Result<RecurrentLayerConfig> {
        let (conv_dim, conv_width) = self.history()?;
        Ok(RecurrentLayerConfig {
            conv_dim,
            conv_width,
            state_dims: Vec::new(),
            conv_dtype: dtype,
            state_dtype: dtype,
        })
    }

    /// Hash each sequence's chunk against the tokens before it and gather its rows:
    /// (batch, seq, rows · width), f32 on `dev` (ple_layer.py:437-456).
    pub(crate) fn embed(
        &self,
        prior: &[Vec<u32>],
        ids: &[Vec<u32>],
        dev: &Device,
    ) -> Result<Tensor> {
        let seq = ids.first().map_or(0, Vec::len);
        if prior.len() != ids.len() || ids.iter().any(|c| c.len() != seq) {
            bail!("n-gram embed needs one prior per sequence and equal chunk lengths");
        }
        let rows: Vec<u32> = prior
            .iter()
            .zip(ids)
            .flat_map(|(p, c)| self.hash.rows(p, c))
            .collect();
        match &self.table {
            Table::Gguf(table) => {
                let rows =
                    Tensor::from_vec(rows, (ids.len(), seq, self.hash.size.len()), &Device::Cpu)?;
                table
                    .embedding(&rows)?
                    .reshape((ids.len(), seq, ()))?
                    .to_device(dev)
            }
            // 16 x 160 bytes of E4M3 per token go up; the device dequantizes and rounds once to
            // bf16, as the served dequant kernel stores f32(fp8) · f32(scale).
            Table::Fp8(table) => {
                let bytes = table.bytes(&rows)?;
                Tensor::from_raw_buffer(
                    &bytes,
                    DType::F8E4M3,
                    &[ids.len(), seq, rows.len() / ids.len().max(1) / seq.max(1) * table.width],
                    &Device::Cpu,
                )?
                .to_device(dev)?
                .to_dtype(DType::F32)?
                .affine(f64::from(table.scale), 0.0)?
                .to_dtype(DType::BF16)
            }
        }
    }

    /// The raw table bytes of `chunk`'s rows (tests).
    #[cfg(test)]
    pub(crate) fn row_bytes(&self, prior: &[u32], chunk: &[u32]) -> Result<Vec<u8>> {
        match &self.table {
            Table::Fp8(t) => t.bytes(&self.hash.rows(prior, chunk)),
            Table::Gguf(_) => bail!("row bytes of a GGUF table"),
        }
    }

    /// The hash's table rows of `chunk` after `prior`.
    pub(crate) fn rows(&self, prior: &[u32], chunk: &[u32]) -> Vec<u32> {
        self.hash.rows(prior, chunk)
    }

    /// The delta for streams `x` (batch, seq, streams, hidden) from embeddings `e` (batch, seq,
    /// width), advancing the conv history in `cache` (ple_layer.py:1155-1188). The delta is f32:
    /// the served graph adds it to the streams inside one kernel and rounds only the sum.
    ///
    /// Numerics follow the served kernels (inductor `triton_red_fused_..._sigmoid_sign_sqrt_...`,
    /// `ple_layer.py:730-830`): key and value are bf16 GEMMs; both norms, their dot product and
    /// the gate stay f32; `gate · value` is stored (and normed from its unrounded square sum);
    /// the norm is stored; the conv (f32 accumulate) is stored, then its silu. With f32
    /// activations every rounding is a no-op.
    pub(crate) fn forward(
        &self,
        x: &Tensor,
        e: &Tensor,
        cache: &mut GdnLayerCache,
    ) -> Result<Tensor> {
        self.forward_probed(x, e, cache, None)
    }

    /// [`Self::forward`], noting each stored stage in `probe` when given (for the goldens).
    pub(crate) fn forward_probed(
        &self,
        x: &Tensor,
        e: &Tensor,
        cache: &mut GdnLayerCache,
        mut probe: Option<&mut Vec<(&'static str, Tensor)>>,
    ) -> Result<Tensor> {
        let mut note = |name: &'static str, t: &Tensor| {
            if let Some(p) = probe.as_mut() {
                p.push((name, t.clone()));
            }
        };
        let (b, s, n, h) = x.dims4()?;
        let dtype = x.dtype();
        let store = |t: &Tensor| -> Result<Tensor> { t.to_dtype(dtype)?.to_dtype(DType::F32) };
        let e = e.to_dtype(dtype)?;
        let key = store(&self.key.forward(&e)?)?.reshape((b, s, n, h))?;
        let value = store(&self.value.forward(&e)?)?;
        note("key", &key);
        note("value", &value);
        let kn = norm_f32(&key, &self.norm_key, self.eps)?;
        let qn = norm_f32(&x.to_dtype(DType::F32)?, &self.norm_query, self.eps)?;
        let scale = (1.0 / (h as f64).sqrt()) as f32;
        let score = (kn * qn)?.sum_keepdim(D::Minus1)?.affine(f64::from(scale), 0.0)?;
        let gate = sigmoid(&(score.sign()? * score.abs()?.maximum(1e-6)?.sqrt()?)?)?;
        let gv_f = gate.broadcast_mul(&value.unsqueeze(2)?)?;
        let gv = store(&gv_f)?;
        note("gv", &gv);
        let ssq = gv_f.sqr()?.sum_keepdim(D::Minus1)?;
        let rstd = ((ssq / h as f64)? + self.eps)?.sqrt()?.recip()?;
        let u = gv
            .broadcast_mul(&rstd)?
            .broadcast_mul(&self.norm_conv.to_dtype(DType::F32)?.reshape((n, h))?)?
            .to_dtype(dtype)?
            .reshape((b, s, n * h))?;
        note("nrm", &u);
        let c = self.conv(&u, cache)?.to_dtype(DType::F32)?;
        note("conv", &c);
        (gv.reshape((b, s, n * h))? + c)?.reshape((b, s, n, h))
    }

    /// silu of the causal depthwise conv over `u` (batch, seq, channels) with taps at t, t-d, ...,
    /// t-(k-1)·d, reading the history in `cache` and leaving the last (k-1)·d inputs there
    /// (ple_layer.py:801-850). The conv accumulates in f32 and is rounded to `u`'s dtype, then
    /// the silu is rounded again, as the eager bf16 `F.conv1d` then `F.silu` store them.
    fn conv(&self, u: &Tensor, cache: &mut GdnLayerCache) -> Result<Tensor> {
        let (b, s, _) = u.dims3()?;
        let (c, rows) = self.history()?;
        let (k, d) = (self.conv.dim(1)?, self.hash.order());
        if cache.conv_state.dims3()? != (b, c, rows) {
            bail!(
                "n-gram conv state is {:?}, expected ({b}, {c}, {rows})",
                cache.conv_state.dims()
            );
        }
        if cache.seqlen_offset == 0 {
            // A new sequence starts from an empty history (:810-815).
            cache.conv_state = cache.conv_state.zeros_like()?;
        }
        cache.trail_conv(u)?;
        // (batch, rows + seq, channels): the carried history, then this chunk.
        let history = Tensor::cat(
            &[&cache.conv_state.to_dtype(u.dtype())?.transpose(1, 2)?, u],
            1,
        )?;
        cache.conv_state = history.narrow(1, s, rows)?.transpose(1, 2)?.contiguous()?;
        cache.seqlen_offset += s;
        // out_t = Σ_j w_j ⊙ history[t + j·d] (:820-826).
        let history = history.to_dtype(DType::F32)?;
        let w = self.conv.to_dtype(DType::F32)?;
        let mut out = history.narrow(1, 0, s)?.broadcast_mul(&w.i((.., 0))?)?;
        for j in 1..k {
            out = (out + history.narrow(1, j * d, s)?.broadcast_mul(&w.i((.., j))?)?)?;
        }
        let out = out.to_dtype(u.dtype())?.to_dtype(DType::F32)?;
        hanzo_nn::ops::silu(&out)?.to_dtype(u.dtype())
    }
}

/// RMS norm over each stream of `x` (.., streams, hidden) with that stream's slice of `w`
/// (streams · hidden), in f32 and not rounded (the served gate kernel keeps it in registers).
fn norm_f32(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let (n, h) = (x.dim(D::Minus2)?, x.dim(D::Minus1)?);
    let f = x.to_dtype(DType::F32)?;
    let rstd = (f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?.recip()?;
    f.broadcast_mul(&rstd)?
        .broadcast_mul(&w.to_dtype(DType::F32)?.reshape((n, h))?)
}

/// RMS norm over each stream of `x` (.., streams, hidden) with that stream's slice of `w`
/// (streams · hidden), in f32, rounded back to `x`'s dtype (ple_layer.py:68-80, 650-653).
fn norm(x: &Tensor, w: &Tensor, eps: f64) -> Result<Tensor> {
    let (n, h) = (x.dim(D::Minus2)?, x.dim(D::Minus1)?);
    let f = x.to_dtype(DType::F32)?;
    let rms = (f.sqr()?.mean_keepdim(D::Minus1)? + eps)?.sqrt()?;
    f.broadcast_div(&rms)?
        .broadcast_mul(&w.reshape((n, h))?)?
        .to_dtype(x.dtype())
}

/// The block in plain f64 loops, one sequence from an empty history.
#[cfg(test)]
mod reference {
    pub(super) struct Weights {
        /// (streams · hidden, width)
        pub(super) key: Vec<f64>,
        /// (hidden, width)
        pub(super) value: Vec<f64>,
        /// (streams · hidden) each
        pub(super) norm_query: Vec<f64>,
        pub(super) norm_key: Vec<f64>,
        pub(super) norm_conv: Vec<f64>,
        /// (streams · hidden, kernel)
        pub(super) conv: Vec<f64>,
    }

    fn rms(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        // ple_layer.py:78-80
        let r = (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
        x.iter().zip(w).map(|(v, w)| v / r * w).collect()
    }

    fn matvec(m: &[f64], x: &[f64]) -> Vec<f64> {
        m.chunks(x.len())
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// Δ (seq, streams · hidden) and the history left behind ((k-1)·d, streams · hidden) for
    /// streams `x` (seq, streams · hidden) and embeddings `e` (seq, width).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn block(
        w: &Weights,
        x: &[f64],
        e: &[f64],
        streams: usize,
        hidden: usize,
        width: usize,
        dilation: usize,
        eps: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let c = streams * hidden;
        let seq = e.len() / width;
        let kernel = w.conv.len() / c;
        let (mut v, mut u) = (vec![0.0; seq * c], vec![0.0; seq * c]);
        for t in 0..seq {
            // ple_layer.py:1171-1181
            let et = &e[t * width..(t + 1) * width];
            let key = matvec(&w.key, et);
            let value = matvec(&w.value, et);
            for st in 0..streams {
                let at = st * hidden..(st + 1) * hidden;
                let k = rms(&key[at.clone()], &w.norm_key[at.clone()], eps);
                let q = rms(&x[t * c..][at.clone()], &w.norm_query[at.clone()], eps);
                let score =
                    k.iter().zip(&q).map(|(a, b)| a * b).sum::<f64>() / (hidden as f64).sqrt();
                let sign = if score > 0.0 {
                    1.0
                } else if score < 0.0 {
                    -1.0
                } else {
                    0.0
                };
                let gate = 1.0 / (1.0 + (-sign * score.abs().max(1e-6).sqrt()).exp());
                let row = t * c + st * hidden;
                for i in 0..hidden {
                    v[row + i] = gate * value[i];
                }
                let normed = rms(&v[row..row + hidden], &w.norm_conv[at], eps);
                u[row..row + hidden].copy_from_slice(&normed);
            }
        }
        // F.conv1d over [zeros((k-1)·d) ++ u], dilation d, then silu (ple_layer.py:816-826).
        let rows = (kernel - 1) * dilation;
        let past = |i: usize, ch: usize| {
            if i < rows {
                0.0
            } else {
                u[(i - rows) * c + ch]
            }
        };
        let mut delta = vec![0.0; seq * c];
        for t in 0..seq {
            for ch in 0..c {
                let a: f64 = (0..kernel)
                    .map(|j| w.conv[ch * kernel + j] * past(t + j * dilation, ch))
                    .sum();
                delta[t * c + ch] = v[t * c + ch] + a / (1.0 + (-a).exp());
            }
        }
        // The next state is the last (k-1)·d inputs (:843-846).
        let history = (seq..seq + rows)
            .flat_map(|i| (0..c).map(move |ch| (i, ch)))
            .map(|(i, ch)| past(i, ch))
            .collect();
        (delta, history)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::io::Cursor;

    use hanzo_ml::quantized::{gguf_file, GgmlDType};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    use super::reference::{block, Weights};
    use super::*;
    use crate::kv_cache::{HybridCache, HybridCacheConfig, HybridLayerCache, HybridLayerType};
    use crate::models::gdn::{forward_pooled, GdnTrail, PoolSlots};

    // A tiny hash small enough to evaluate by hand: order 3, two heads per order.
    const EOS: u32 = 9;
    const MULT: [u64; 3] = [3, 5, 7];
    const SIZE: [u64; 4] = [11, 13, 17, 19];
    const OFFSET: [u64; 4] = [0, 11, 24, 41];
    const HEADS: u32 = 2;
    // The block around it: 3 streams of 8, table rows of 3 (so 12-wide embeddings), kernel 4.
    const STREAMS: usize = 3;
    const HIDDEN: usize = 8;
    const ROW: usize = 3;
    const WIDTH: usize = 4 * ROW;
    const KERNEL: usize = 4;
    const EPS: f64 = 1e-6;

    fn metadata(
        eos: u32,
        heads: u32,
        mult: &[u64],
        offset: &[u64],
        size: &[u64],
    ) -> HashMap<String, gguf_file::Value> {
        use gguf_file::Value;
        let u64s = |v: &[u64]| Value::Array(v.iter().map(|&x| Value::U64(x)).collect());
        HashMap::from([
            (
                "qwen4exp.ple.ngram_size".into(),
                Value::U32(mult.len() as u32),
            ),
            ("qwen4exp.ple.heads_per_ngram".into(), Value::U32(heads)),
            ("qwen4exp.ple.eos_token_id".into(), Value::U32(eos)),
            ("qwen4exp.ple.layer_multipliers".into(), u64s(mult)),
            ("qwen4exp.ple.head_offsets".into(), u64s(offset)),
            ("qwen4exp.ple.head_vocab_sizes".into(), u64s(size)),
        ])
    }

    fn hash_of(md: &HashMap<String, gguf_file::Value>) -> Result<Hash> {
        Hash::from_gguf(&ContentMetadata {
            path_prefix: "qwen4exp",
            metadata: md,
        })
    }

    fn tiny() -> Result<Hash> {
        hash_of(&metadata(EOS, HEADS, &MULT, &OFFSET, &SIZE))
    }

    /// The header of the Qwen3.8-Flash-Next UD-IQ4_XS GGUF.
    fn real() -> Result<Hash> {
        hash_of(&metadata(
            248044,
            8,
            &[23703573157769, 20109073645365, 8052911324071],
            &[
                0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
                160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114,
                300001275,
            ],
            &[
                20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
                20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
            ],
        ))
    }

    /// Row ids worked out from the formula. [2, 4] after an empty prior reads EOS EOS 2 4:
    ///   t=0: 2·3 ^ 9·5 = 43 -> 43%11, 43%13+11;  43 ^ 9·7 = 20 -> 20%17+24, 20%19+41
    ///   t=1: 4·3 ^ 2·5 = 6  -> 6%11,  6%13+11;   6 ^ 9·7 = 57  -> 57%17+24, 57%19+41
    #[test]
    fn hash_rows_by_hand() -> Result<()> {
        assert_eq!(tiny()?.rows(&[], &[2, 4]), [10, 15, 27, 42, 6, 17, 30, 41]);
        Ok(())
    }

    /// An EOS cuts off the tokens before it for the positions after it, not for itself.
    #[test]
    fn hash_eos_rule() -> Result<()> {
        let hash = tiny()?;
        // [1, 9] at t=1 reads EOS 1 9, with the 1 intact:
        //   9·3 ^ 1·5 = 30 -> 30%11, 30%13+11;  30 ^ 9·7 = 33 -> 33%17+24, 33%19+41
        assert_eq!(hash.rows(&[], &[1, 9])[4..], [8, 15, 40, 55]);
        // After 3 1 9, the 4 reads EOS EOS 4, exactly as at the start of a sequence:
        //   4·3 ^ 9·5 = 33 -> 33%11, 33%13+11;  33 ^ 9·7 = 30 -> 30%17+24, 30%19+41
        let fresh = hash.rows(&[], &[4]);
        assert_eq!(fresh, [0, 18, 37, 52]);
        assert_eq!(hash.rows(&[3], &[1, 9, 4])[8..], fresh[..]);
        Ok(())
    }

    /// The shipped constants load, chain, are prime, stay inside the 320001536-row table, and hash
    /// [9707, 11, 1879] as a plain transcription of ple_layer.py:336-435 does; for instance
    /// row 0 of the last token is (1879·m0 ^ 11·m1) % 20000003 = 6380558.
    #[test]
    fn hash_real_constants() -> Result<()> {
        let hash = real()?;
        assert!(hash
            .size
            .iter()
            .all(|&p| p > 1 && (2..).take_while(|d| d * d <= p).all(|d| p % d != 0)));
        assert!(hash.offset[15] + hash.size[15] <= 320001536);
        assert_eq!(
            hash.rows(&[], &[9707, 11, 1879])[32..],
            [
                6380558, 26411572, 56460672, 78566983, 94693008, 106742196, 124822692, 148942556,
                164226950, 190352573, 210933682, 238908951, 242182004, 265475238, 299910982,
                312121804
            ]
        );
        // Tables that do not sit end to end are refused.
        let mut broken = OFFSET;
        broken[2] += 1;
        assert!(hash_of(&metadata(EOS, HEADS, &MULT, &broken, &SIZE)).is_err());
        Ok(())
    }

    /// Cutting a sequence into (prior, chunk) anywhere gives the rows of one pass, whether the
    /// prior is the whole prefix or just its last two tokens.
    #[test]
    fn hash_split_anywhere() -> Result<()> {
        let hash = real()?;
        let mut rng = StdRng::seed_from_u64(7);
        let seq: Vec<u32> = (0..40)
            .map(|_| {
                if rng.random_range(0..4) == 0 {
                    248044
                } else {
                    rng.random_range(0..248320)
                }
            })
            .collect();
        let whole = hash.rows(&[], &seq);
        let per = hash.size.len();
        for at in 0..=seq.len() {
            let tail = &whole[at * per..];
            assert_eq!(hash.rows(&seq[..at], &seq[at..]), tail, "split at {at}");
            assert_eq!(
                hash.rows(&seq[at.saturating_sub(2)..at], &seq[at..]),
                tail,
                "split at {at}, two-token prior"
            );
        }
        Ok(())
    }

    fn uniform(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| rng.random_range(lo..hi)).collect()
    }

    fn wide(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| f64::from(x)).collect()
    }

    /// A GGUF holding the tiny block at `blk.1`, loaded the way a model loads it, with the same
    /// weights for the reference.
    fn fixture(seed: u64) -> Result<(Ngram, Weights)> {
        let dev = Device::Cpu;
        let mut rng = StdRng::seed_from_u64(seed);
        let c = STREAMS * HIDDEN;
        let rows = (OFFSET[3] + SIZE[3]) as usize;
        // bf16-exact taps, so the load-time bf16 cast is lossless and the reference sees them too.
        let conv = Tensor::from_vec(uniform(&mut rng, c * KERNEL, -0.5, 0.5), (c, KERNEL), &dev)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?;
        let tensors = [
            (
                "per_layer_token_embd",
                Tensor::from_vec(uniform(&mut rng, rows * ROW, -1.0, 1.0), (rows, ROW), &dev)?,
            ),
            (
                "blk.1.ple_key",
                Tensor::from_vec(uniform(&mut rng, c * WIDTH, -0.5, 0.5), (c, WIDTH), &dev)?,
            ),
            (
                "blk.1.ple_value",
                Tensor::from_vec(
                    uniform(&mut rng, HIDDEN * WIDTH, -0.5, 0.5),
                    (HIDDEN, WIDTH),
                    &dev,
                )?,
            ),
            (
                "blk.1.ple_norm_query",
                Tensor::from_vec(uniform(&mut rng, c, 0.7, 1.3), c, &dev)?,
            ),
            (
                "blk.1.ple_norm_key",
                Tensor::from_vec(uniform(&mut rng, c, 0.7, 1.3), c, &dev)?,
            ),
            (
                "blk.1.ple_norm_conv",
                Tensor::from_vec(uniform(&mut rng, c, 0.7, 1.3), c, &dev)?,
            ),
            ("blk.1.ple_conv1d", conv),
        ];
        let flat = |i: usize| -> Result<Vec<f64>> {
            Ok(wide(&tensors[i].1.flatten_all()?.to_vec1::<f32>()?))
        };
        let weights = Weights {
            key: flat(1)?,
            value: flat(2)?,
            norm_query: flat(3)?,
            norm_key: flat(4)?,
            norm_conv: flat(5)?,
            conv: flat(6)?,
        };

        let quantized = tensors
            .iter()
            .map(|(name, t)| -> Result<(String, QTensor)> {
                Ok((
                    format!("{name}.weight"),
                    QTensor::quantize(t, GgmlDType::F32)?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        // Content wants an architecture it knows; the block reads only the keys and tensors above.
        let arch = gguf_file::Value::String("qwen35moe".into());
        let md = metadata(EOS, HEADS, &MULT, &OFFSET, &SIZE);
        let mut kv: Vec<(&str, &gguf_file::Value)> = vec![("general.architecture", &arch)];
        kv.extend(md.iter().map(|(k, v)| (k.as_str(), v)));
        let named: Vec<(&str, &QTensor)> = quantized.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut file = Cursor::new(Vec::new());
        gguf_file::write(&mut file, &kv, &named)?;

        let mut file = Cursor::new(file.into_inner());
        let mut readers = [&mut file];
        let mut ct = Content::from_readers(&mut readers)?;
        let hash = Hash::from_gguf(&ContentMetadata {
            path_prefix: "qwen4exp",
            metadata: ct.get_metadata(),
        })?;
        Ok((Ngram::load(&mut ct, hash, 1, EPS, &dev)?, weights))
    }

    fn fresh(ngram: &Ngram, batch: usize) -> Result<GdnLayerCache> {
        let (c, rows) = ngram.history()?;
        Ok(GdnLayerCache {
            conv_state: Tensor::zeros((batch, c, rows), DType::F32, &Device::Cpu)?,
            recurrent_state: Tensor::zeros(batch, DType::F32, &Device::Cpu)?,
            seqlen_offset: 0,
            trail: None,
        })
    }

    /// Streams (batch, seq, STREAMS, HIDDEN) and embeddings (batch, seq, WIDTH).
    fn inputs(seed: u64, batch: usize, seq: usize) -> Result<(Tensor, Tensor)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let x = uniform(&mut rng, batch * seq * STREAMS * HIDDEN, -2.0, 2.0);
        let e = uniform(&mut rng, batch * seq * WIDTH, -1.0, 1.0);
        Ok((
            Tensor::from_vec(x, (batch, seq, STREAMS, HIDDEN), &Device::Cpu)?,
            Tensor::from_vec(e, (batch, seq, WIDTH), &Device::Cpu)?,
        ))
    }

    /// max |got - want| / max |want|
    fn gap(got: &Tensor, want: &[f64]) -> Result<f64> {
        let got = got.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0f64, |m, w| m.max(w.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (g, w)| m.max((f64::from(*g) - w).abs()));
        Ok(worst / scale)
    }

    /// The block matches the f64 transcription of vLLM, delta and the history it leaves.
    #[test]
    fn block_matches_reference() -> Result<()> {
        let (ngram, weights) = fixture(1)?;
        let (batch, seq) = (2, 13);
        let (x, e) = inputs(2, batch, seq)?;
        let mut cache = fresh(&ngram, batch)?;
        let delta = ngram.forward(&x, &e, &mut cache)?;
        assert_eq!(delta.dims(), &[batch, seq, STREAMS, HIDDEN]);
        assert_eq!(cache.seqlen_offset, seq);
        for i in 0..batch {
            let xi = wide(&x.i(i)?.flatten_all()?.to_vec1::<f32>()?);
            let ei = wide(&e.i(i)?.flatten_all()?.to_vec1::<f32>()?);
            let (want, history) = block(&weights, &xi, &ei, STREAMS, HIDDEN, WIDTH, 3, EPS);
            let d = gap(&delta.i(i)?, &want)?;
            assert!(d < 1e-5, "sequence {i}: delta off by {d:e}");
            // The cache holds (channels, rows); the reference lists rows of channels.
            let h = gap(&cache.conv_state.i(i)?.t()?, &history)?;
            assert!(h < 1e-5, "sequence {i}: history off by {h:e}");
        }
        Ok(())
    }

    /// Chunks carried through the cache, one of them a single decode token, give the one-pass delta.
    #[test]
    fn block_chunked_matches_whole() -> Result<()> {
        let (ngram, _) = fixture(3)?;
        let (batch, seq) = (2, 13);
        let (x, e) = inputs(4, batch, seq)?;
        let mut cache = fresh(&ngram, batch)?;
        let whole = ngram.forward(&x, &e, &mut cache)?;
        let mut chunked = fresh(&ngram, batch)?;
        let mut parts = Vec::new();
        let mut at = 0;
        for len in [5, 1, 7] {
            parts.push(ngram.forward(
                &x.narrow(1, at, len)?,
                &e.narrow(1, at, len)?,
                &mut chunked,
            )?);
            at += len;
        }
        let want = wide(&whole.flatten_all()?.to_vec1::<f32>()?);
        let d = gap(&Tensor::cat(&parts, 1)?, &want)?;
        assert!(d < 1e-5, "chunked delta off by {d:e}");
        let state = wide(&cache.conv_state.flatten_all()?.to_vec1::<f32>()?);
        let h = gap(&chunked.conv_state, &state)?;
        assert!(h < 1e-5, "chunked history off by {h:e}");
        Ok(())
    }

    /// A continuing forward that keeps a trail notes, after each position, the history a forward
    /// stopping there would leave, which is what a rewind restores.
    #[test]
    fn block_trail_is_the_history_after_each_position() -> Result<()> {
        let (ngram, _) = fixture(5)?;
        let (batch, head, tail) = (2, 5, 8);
        let (x, e) = inputs(6, batch, head + tail)?;
        let mut cache = fresh(&ngram, batch)?;
        ngram.forward(&x.narrow(1, 0, head)?, &e.narrow(1, 0, head)?, &mut cache)?;
        cache.trail = Some(GdnTrail::default());
        ngram.forward(
            &x.narrow(1, head, tail)?,
            &e.narrow(1, head, tail)?,
            &mut cache,
        )?;
        let trail = cache
            .trail
            .expect("the forward keeps the trail it was asked for");
        assert_eq!(trail.conv.len(), tail);
        for (t, noted) in trail.conv.iter().enumerate() {
            let mut stop = fresh(&ngram, batch)?;
            let len = head + t + 1;
            ngram.forward(&x.narrow(1, 0, len)?, &e.narrow(1, 0, len)?, &mut stop)?;
            let want = wide(&stop.conv_state.flatten_all()?.to_vec1::<f32>()?);
            let h = gap(noted, &want)?;
            assert!(h < 1e-5, "trail after position {t} off by {h:e}");
        }
        Ok(())
    }

    /// A cache with one attention layer and the block's side pool after it.
    fn side_cache(ngram: &Ngram) -> Result<HybridCache> {
        let cfg = HybridCacheConfig {
            layer_types: vec![HybridLayerType::Attention],
            max_seq_len: 64,
            pools: vec![ngram.pool(DType::F32)?],
        };
        HybridCache::new(cfg, &Device::Cpu)
    }

    /// One forward of the block through the side pool, as the model runs it.
    fn pooled(
        ngram: &Ngram,
        cache: &mut HybridCache,
        slot: usize,
        (x, e): (&Tensor, &Tensor),
        trail: bool,
    ) -> Result<Tensor> {
        let side = cache.num_layers();
        let pool = cache
            .get_mut(side)
            .and_then(HybridLayerCache::as_recurrent_pool_mut)
            .expect("the side pool");
        let offset = pool.get_seqlen_offset(slot);
        forward_pooled(pool, PoolSlots::One { slot, offset }, side, trail, |c| {
            ngram.forward(x, e, c)
        })
    }

    /// In its side pool the block rewinds and restores like a GDN layer. A verify keeps the conv
    /// trail; rewinding the rejected drafts and decoding on gives what decoding the accepted
    /// tokens one at a time gives, and so does a snapshot of the prompt restored into another slot.
    #[test]
    fn block_rewinds_and_restores_in_a_side_pool() -> Result<()> {
        const VERIFY: usize = 4;
        let (prompt, kept) = (6, 2);
        let (ngram, _) = fixture(9)?;
        let (x, e) = inputs(10, 1, prompt + VERIFY)?;
        let (wrong_x, wrong_e) = inputs(11, 1, VERIFY - kept)?;
        let at = |t: usize, len: usize| -> Result<(Tensor, Tensor)> {
            Ok((x.narrow(1, t, len)?, e.narrow(1, t, len)?))
        };
        let (px, pe) = at(0, prompt)?;

        let mut plain = side_cache(&ngram)?;
        let slot = plain.allocate_seq().expect("a slot");
        pooled(&ngram, &mut plain, slot, (&px, &pe), false)?;
        let mut want = Vec::with_capacity(VERIFY);
        for t in prompt..prompt + VERIFY {
            let (xt, et) = at(t, 1)?;
            let d = pooled(&ngram, &mut plain, slot, (&xt, &et), false)?;
            want.push(wide(&d.flatten_all()?.to_vec1::<f32>()?));
        }

        let mut spec = side_cache(&ngram)?;
        let slot = spec.allocate_seq().expect("a slot");
        pooled(&ngram, &mut spec, slot, (&px, &pe), false)?;
        let snapshot = spec.snapshot_recurrent_state(slot)?;
        let (ax, ae) = at(prompt, kept)?;
        let vx = Tensor::cat(&[&ax, &wrong_x], 1)?;
        let ve = Tensor::cat(&[&ae, &wrong_e], 1)?;
        let verified = pooled(&ngram, &mut spec, slot, (&vx, &ve), true)?;
        for (t, want) in want.iter().enumerate().take(kept) {
            let d = gap(&verified.narrow(1, t, 1)?, want)?;
            assert!(d < 1e-5, "accepted draft {t} off by {d:e}");
        }
        spec.rewind_recurrent(slot, VERIFY - kept)?;
        assert_eq!(spec.recurrent_offset(slot), Some(prompt + kept));
        for (t, want) in want.iter().enumerate().skip(kept) {
            let (xt, et) = at(prompt + t, 1)?;
            let d = gap(&pooled(&ngram, &mut spec, slot, (&xt, &et), false)?, want)?;
            assert!(d < 1e-5, "token {t} after the rewind off by {d:e}");
        }

        let other = spec.allocate_seq().expect("a second slot");
        spec.restore_recurrent_state(other, &snapshot)?;
        assert_eq!(spec.recurrent_offset(other), Some(prompt));
        for (t, want) in want.iter().enumerate() {
            let (xt, et) = at(prompt + t, 1)?;
            let d = gap(&pooled(&ngram, &mut spec, other, (&xt, &et), false)?, want)?;
            assert!(d < 1e-5, "token {t} after the restore off by {d:e}");
        }
        Ok(())
    }

    /// The embedding of a chunk is its hashed rows of the table, laid end to end.
    #[test]
    fn embed_gathers_hashed_rows() -> Result<()> {
        let (ngram, _) = fixture(7)?;
        let prior = vec![vec![5, 1], vec![]];
        let ids = vec![vec![2, 9, 4], vec![7, 7, 3]];
        let e = ngram.embed(&prior, &ids, &Device::Cpu)?;
        assert_eq!(e.dims(), &[2, 3, WIDTH]);
        let Table::Gguf(table) = &ngram.table else {
            panic!("a GGUF table")
        };
        let table = table.dequantize(&Device::Cpu)?;
        for (i, (p, c)) in prior.iter().zip(&ids).enumerate() {
            let rows = Tensor::new(ngram.hash.rows(p, c), &Device::Cpu)?;
            let want = table.index_select(&rows, 0)?.reshape((3, WIDTH))?;
            let want = wide(&want.flatten_all()?.to_vec1::<f32>()?);
            assert_eq!(gap(&e.i(i)?, &want)?, 0.0, "sequence {i}");
        }
        Ok(())
    }

    /// A table whose data starts 2 bytes into the data section (an odd 2-byte offset): rows come
    /// back byte-exact, and the device dequant is f32(fp8) · f32(scale) rounded once.
    #[test]
    fn fp8_table_reads_unaligned_rows() -> Result<()> {
        let (rows, width, shards) = (5usize, 6usize, 3usize);
        let bytes: Vec<u8> = (0..rows * width * shards).map(|i| (i * 37 % 251) as u8 & 0x7e).collect();
        let mut header = serde_json::Map::new();
        // a 2-byte BF16 scale first, so the shards start at data offset 2
        header.insert(
            "t.weight_scale".into(),
            serde_json::json!({"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}),
        );
        for sh in 0..shards {
            let start = 2 + sh * rows * width;
            header.insert(
                format!("t.shard_{sh}.weight"),
                serde_json::json!({"dtype": "F8_E4M3", "shape": [rows, width], "data_offsets": [start, start + rows * width]}),
            );
        }
        let mut json = serde_json::to_vec(&header).map_err(hanzo_ml::Error::wrap)?;
        // an odd header length puts the data section itself off 8-byte alignment too
        while (8 + json.len()) % 2 == 0 {
            json.push(b' ');
        }
        let scale = half::bf16::from_f32(0.0123);
        let mut file = (json.len() as u64).to_le_bytes().to_vec();
        file.extend(&json);
        file.extend(scale.to_le_bytes());
        file.extend(&bytes);
        let path = std::env::temp_dir().join(format!("hanzo-ple-table-{}.safetensors", std::process::id()));
        std::fs::write(&path, &file).map_err(hanzo_ml::Error::wrap)?;
        let table = Fp8Table::open(&[path], "t")?;
        let picks = [0u32, 4, 5, 9, 14, 7];
        let got = table.bytes(&picks)?;
        for (i, &r) in picks.iter().enumerate() {
            let at = r as usize * width;
            assert_eq!(&got[i * width..(i + 1) * width], &bytes[at..at + width], "row {r}");
        }
        assert_eq!(table.scale, scale.to_f32());
        let e = Tensor::from_raw_buffer(&got, DType::F8E4M3, &[picks.len(), width], &Device::Cpu)?
            .to_dtype(DType::F32)?
            .affine(f64::from(table.scale), 0.0)?
            .to_dtype(DType::BF16)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (i, b) in got.iter().enumerate() {
            let want = half::bf16::from_f32(float8::F8E4M3::from_bits(*b).to_f32() * scale.to_f32()).to_f32();
            assert_eq!(e[i].to_bits(), want.to_bits(), "element {i}");
        }
        Ok(())
    }
}
