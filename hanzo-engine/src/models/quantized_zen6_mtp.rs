//! The multi-token-prediction head of Qwen3.8-Flash-Next, from the GGUF published beside the model
//! (`mtp-Qwen3.8-Flash-Next-*.gguf`: arch `qwen4exp`, one block at `blk.{depth}`).
//!
//! From the target's hyper-connection streams `h_t` (the last block's output, before the output
//! mixer) and the token `x_{t+1}`, it predicts `x_{t+2}`:
//!
//! ```text
//! r_s    = eh_proj [enorm(embed(x_{t+1})) | hnorm_s(h_t,s)]    per stream s, [2h] -> [h]
//! r      = ffn_hc(attn_hc(r, attention), moe)                   one zen6 attention block
//! logits = lm_head(mix(r))                                      its own mixer is the output norm
//! ```
//!
//! `r` is the next draft step's `h`. A `shared` file carries neither the token embedding nor the
//! output projection and borrows the target's; a self-contained file carries its own.
//!
//! The block's attention keeps its own KV over the whole sequence, as it was trained. Every target
//! forward writes the K/V of each pair it completes, and a draft step drops whatever sits at or
//! past its own position, so pairs written for rejected drafts never reach a later draft. A pair
//! `(h_t, x_{t+1})` sits at position `t`; shifting every pair one position later changes no
//! distance.
//! Attention is dense, as in the reference: the head's indexer tensors are not read.

use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::{Arc, Mutex};

use hanzo_ml::{DType, Device, Result, Tensor, D};
use hanzo_nn::{Embedding, Module};
use hanzo_quant::QuantMethod;

use crate::attention::AttentionMask;
use crate::gguf::Content;
use crate::kv_cache::{KvCache, NormalCache};
use crate::layers::{QRmsNorm, Qwen3VLRotaryEmbedding};
use crate::models::hyper::{expand, Branch, Mixer, Norm};
use crate::models::quantized_qwen3_5_moe::{
    gguf_qmm, text_mrope, verify_arch, FusedMoe, GatedFullAttention, PropsGGUF,
};
use crate::models::quantized_zen6::{ModelWeights, ARCH};
use crate::models::qwen3_5_mtp::{
    default_n_predict, AnchorPositions, MtpOut, MtpStep, Qwen3_5MtpProposer,
};
use crate::pipeline::extract_logits;
use crate::speculative::target::SharedLayer;
use crate::speculative::{MtpConfig, SelfSpeculative, SpeculativeProposer, SpeculativeSharedHeads};
use crate::utils::gguf_metadata::ContentMetadata;

/// The block at `blk.{depth}` with its entry and exit.
struct Head {
    enorm: QRmsNorm,
    hnorm: Norm,
    eh_proj: Arc<dyn QuantMethod>,
    attn: GatedFullAttention,
    attn_hc: Branch,
    ffn_hc: Branch,
    moe: FusedMoe,
    mixer: Mixer,
    rotary: Arc<Qwen3VLRotaryEmbedding>,
    streams: usize,
}

/// A head file: the block, its width, and the token embedding and output projection a
/// self-contained file carries, each `[vocab, hidden]`.
struct HeadFile {
    head: Head,
    hidden: usize,
    embed: Option<Tensor>,
    output: Option<(Arc<dyn QuantMethod>, usize)>,
}

impl Head {
    /// The head in `ct`, rotating as the target's attention does.
    fn load<R: Read + Seek>(
        ct: &mut Content<'_, R>,
        rotary: Arc<Qwen3VLRotaryEmbedding>,
        device: &Device,
        dtype: DType,
    ) -> Result<HeadFile> {
        let meta = ct.get_metadata();
        verify_arch(meta, &[ARCH])?;
        let md = ContentMetadata {
            path_prefix: ARCH,
            metadata: meta,
        };
        let props = PropsGGUF::try_from(&md, true)?;
        let streams = md
            .get_value::<u32>("hyper_connection.count")
            .map_err(|e| hanzo_ml::Error::Msg(format!("{e}")))? as usize;
        match props.nextn_predict_layers {
            1 => {}
            0 => hanzo_ml::bail!("this GGUF carries no MTP head (nextn_predict_layers = 0)"),
            n => hanzo_ml::bail!("the zen6 MTP head is one block; this GGUF declares {n}"),
        }
        // The converter counts the head in `block_count`; it sits after the transformer.
        let prefix = format!("blk.{}", props.block_count);
        let nextn = |name: &str| format!("{prefix}.nextn.{name}");
        let eps = props.rms_norm_eps;
        let head = Self {
            enorm: QRmsNorm::new(ct.tensor(&nextn("enorm.weight"), device)?, eps)?,
            hnorm: Norm::from_gguf(ct, &nextn("hnorm.weight"), streams, eps, device)?,
            eh_proj: gguf_qmm(ct.tensor(&nextn("eh_proj.weight"), device)?)?,
            // The head keeps its own KV, so it takes no slot in the target's paged cache.
            attn: GatedFullAttention::load(
                ct,
                &prefix,
                &props,
                rotary.clone(),
                None,
                device,
                dtype,
            )?,
            attn_hc: Branch::from_gguf(ct, &format!("{prefix}.hc_attn"), streams, eps, device)?,
            ffn_hc: Branch::from_gguf(ct, &format!("{prefix}.hc_ffn"), streams, eps, device)?,
            moe: FusedMoe::from_gguf(ct, &prefix, device, props.num_experts_per_tok)?,
            mixer: Mixer::from_gguf(ct, &nextn("hc_head"), streams, eps, device)?,
            rotary,
            streams,
        };
        // Held as the target holds its own: the table dequantized, the projection quantized.
        let embed = if ct.has_tensor("token_embd.weight") {
            Some(ct.tensor("token_embd.weight", device)?.dequantize(device)?)
        } else {
            None
        };
        let output = if ct.has_tensor("output.weight") {
            let q = ct.tensor("output.weight", device)?;
            let vocab = q.shape().dims2()?.0;
            Some((gguf_qmm(q)?, vocab))
        } else {
            None
        };
        Ok(HeadFile {
            head,
            hidden: props.embedding_length,
            embed,
            output,
        })
    }

    /// `r_s = eh_proj [enorm(e) | hnorm_s(h_s)]` for embeddings `[b, r, h]` and target streams
    /// `[b, r, n·h]`, as `[b, r, n, h]`. The projection runs per stream: pooling the streams first
    /// would discard the residual they carry.
    fn entry(&self, embeds: &Tensor, hidden: &Tensor) -> Result<Tensor> {
        let (b, r, h) = embeds.dims3()?;
        let n = self.streams;
        let e = expand(&self.enorm.forward(&embeds.to_dtype(DType::F32)?)?, n)?;
        let hn = self
            .hnorm
            .forward(&hidden.reshape((b, r, n, h))?.to_dtype(DType::F32)?)?;
        let cat = Tensor::cat(&[&e, &hn], D::Minus1)?;
        self.eh_proj
            .forward(&cat.reshape((b, r * n, 2 * h))?)?
            .reshape((b, r, n, h))
    }

    /// Appends to `kv` the keys and values of the pairs `(hidden[i], embeds[i])` at positions
    /// `first..`, as `[1, r, ..]` inputs. The block's output is never read for these, so nothing
    /// past the keys and values runs.
    fn write(
        &self,
        embeds: &Tensor,
        hidden: &Tensor,
        first: usize,
        kv: &mut KvCache,
    ) -> Result<()> {
        let u = self.attn_hc.input(&self.entry(embeds, hidden)?)?;
        let cos_sin = text_mrope(
            &self.rotary,
            u.device(),
            &[first],
            u.dim(1)?,
            u.dtype(),
            None,
        )?;
        let (_, k, v, _) = self.attn.project(&u, &cos_sin)?;
        kv.append(&k, &v)?;
        Ok(())
    }
}

/// One sequence's side of the head.
struct Track {
    /// Row `i` holds the pair at position `base + i`.
    kv: KvCache,
    base: usize,
    /// The target streams of the sequence's last forward, `[rows, n·h]` from that position.
    last: Option<(usize, Tensor)>,
}

impl Track {
    fn new(max_seq_len: usize) -> Self {
        // The cache grows a block at a time and refuses to outgrow its bound, so the bound is a
        // whole number of blocks.
        let block = NormalCache::CACHE_GROW_SIZE;
        Self {
            kv: KvCache::new_normal(2, max_seq_len.next_multiple_of(block), block),
            base: 0,
            last: None,
        }
    }

    /// Drops every pair at or past `pos`, so the next one lands there. A cache that cannot reach
    /// `pos`, starting past it or ending before it, restarts there.
    fn seek(&mut self, pos: usize) -> Result<()> {
        let len = self.kv.current_seq_len();
        if self.base <= pos && pos <= self.base + len {
            self.kv.set_len(pos - self.base)
        } else {
            self.kv.reset();
            self.base = pos;
            Ok(())
        }
    }

    /// The target streams at `pos` from the last forward, `[1, n·h]`, if it produced them.
    fn streams(&self, pos: usize) -> Result<Option<Tensor>> {
        match &self.last {
            Some((start, rows)) if *start <= pos && pos < start + rows.dim(0)? => {
                rows.narrow(0, pos - start, 1).map(Some)
            }
            _ => Ok(None),
        }
    }
}

/// The head beside its target: the block, each sequence's track, and what the target's last
/// forward left for the next draft.
pub(crate) struct Mtp {
    head: Head,
    embed: SharedLayer,
    tracks: HashMap<usize, Track>,
    /// The sequences the next forward runs, in batch order.
    forward: Vec<usize>,
    /// The streams of the rows the last forward took logits from, `[batch, rows, n·h]`, and the
    /// position of each row.
    spec: Option<(Tensor, Vec<Vec<usize>>)>,
    anchors: AnchorPositions,
    max_seq_len: usize,
}

/// Where a target keeps its head: empty until one is attached.
pub(crate) type Slot = Arc<Mutex<Option<Mtp>>>;

impl Mtp {
    /// The head beside a target whose positions run to `max_seq_len`, embedding tokens by
    /// `embed`, with no sequence seen yet.
    fn new(head: Head, embed: SharedLayer, anchors: AnchorPositions, max_seq_len: usize) -> Self {
        Self {
            head,
            embed,
            tracks: HashMap::new(),
            forward: Vec::new(),
            spec: None,
            anchors,
            max_seq_len,
        }
    }

    /// Takes in one target forward: tokens `ids` `[b, s]` from positions `offsets`, and the
    /// streams `x` `[b, s, n, h]` the last block left. Each row writes the pairs it completes into
    /// its sequence's track.
    fn observe(
        &mut self,
        ids: &Tensor,
        x: &Tensor,
        offsets: &[usize],
        context_lens: &[(usize, usize)],
    ) -> Result<()> {
        let seqs = std::mem::take(&mut self.forward);
        let (b, s, n, h) = x.dims4()?;
        let x = x.reshape((b, s, n * h))?;
        let positions = context_lens
            .iter()
            .zip(offsets)
            .map(|(&(start, len), &offset)| (offset + start..offset + start + len).collect())
            .collect();
        self.spec = Some((extract_logits(&x, context_lens.to_vec())?, positions));
        // A forward with no sequence named for each row feeds no track. Each sequence's next
        // forward then finds its track out of reach and restarts it there.
        if seqs.len() != b {
            return Ok(());
        }
        let embeds = (self.embed)(ids)?;
        let Self {
            head,
            tracks,
            max_seq_len,
            ..
        } = self;
        for (row, seq) in seqs.iter().enumerate() {
            let at = offsets[row];
            let streams = x.get(row)?;
            let track = tracks
                .entry(*seq)
                .or_insert_with(|| Track::new(*max_seq_len));
            // Token `at` pairs with the streams at `at - 1`, which the previous forward left when
            // it ran that position. Without them the first pair here is the next token's.
            let prev = match at.checked_sub(1) {
                Some(p) => track.streams(p)?,
                None => None,
            };
            let skip = usize::from(prev.is_none());
            let count = s - skip;
            if count > 0 {
                let hidden = match prev {
                    Some(prev) if s == 1 => prev,
                    Some(prev) => Tensor::cat(&[&prev, &streams.narrow(0, 0, s - 1)?], 0)?,
                    None => streams.narrow(0, 0, count)?,
                };
                let tokens = embeds.get(row)?.narrow(0, skip, count)?;
                let first = at + skip - 1;
                track.seek(first)?;
                head.write(
                    &tokens.unsqueeze(0)?,
                    &hidden.unsqueeze(0)?,
                    first,
                    &mut track.kv,
                )?;
            }
            track.last = Some((at, streams));
        }
        Ok(())
    }

    /// One draft step: row `i` of `[b, 1, ..]` inputs for sequence `seqs[i]`, at the position
    /// `positions` holds for it.
    fn draft(
        &mut self,
        seqs: &[usize],
        embeds: &Tensor,
        hidden: &Tensor,
        positions: &Tensor,
    ) -> Result<MtpOut> {
        let b = embeds.dim(0)?;
        let at: Vec<u32> = positions.get(0)?.flatten_all()?.to_vec1()?;
        let Self {
            head,
            tracks,
            max_seq_len,
            ..
        } = self;
        let res = head.entry(embeds, hidden)?;
        let (cos, sin) = head.rotary.compute_cos_sin(positions, res.dtype())?;
        let res = head.attn_hc.apply(&res, |u| {
            let mut ys = Vec::with_capacity(b);
            for (i, seq) in seqs.iter().enumerate() {
                let track = tracks
                    .entry(*seq)
                    .or_insert_with(|| Track::new(*max_seq_len));
                track.seek(at[i] as usize)?;
                let cos_sin = (cos.narrow(0, i, 1)?, sin.narrow(0, i, 1)?);
                // Every cached row precedes this query, so the step needs no mask.
                ys.push(head.attn.forward(
                    &u.narrow(0, i, 1)?,
                    &AttentionMask::None,
                    &cos_sin,
                    &mut track.kv,
                    None,
                )?);
            }
            Tensor::cat(&ys, 0)
        })?;
        let res = head.ffn_hc.apply(&res, |v| head.moe.forward(v))?;
        Ok(MtpOut {
            head: head.mixer.mix(&head.mixer.norm(&res)?)?,
            carry: res.reshape((b, 1, ()))?,
        })
    }
}

/// The head as the chain drives it: a handle on its target's slot.
struct Drafter {
    slot: Slot,
    device: Device,
    /// The target streams' dtype, which is its embedding's.
    dtype: DType,
}

impl MtpStep for Drafter {
    fn step(
        &mut self,
        seqs: &[usize],
        input_embeds: &Tensor,
        target_hidden: &Tensor,
        positions: &Tensor,
    ) -> Result<MtpOut> {
        let mut slot = self
            .slot
            .lock()
            .map_err(|_| hanzo_ml::Error::msg("zen6 MTP state poisoned"))?;
        let mtp = slot
            .as_mut()
            .ok_or_else(|| hanzo_ml::Error::msg("the zen6 MTP head was detached"))?;
        mtp.draft(seqs, input_embeds, target_hidden, positions)
    }

    /// A chain needs no reset: each step drops what sits at or past its own position.
    fn reset(&mut self) {}

    fn retain(&mut self, live: &[usize]) {
        if let Ok(mut slot) = self.slot.lock() {
            if let Some(mtp) = slot.as_mut() {
                mtp.tracks.retain(|seq, _| live.contains(seq));
            }
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

impl ModelWeights {
    /// Names the sequences the next forward runs, in batch order, so each row feeds its own
    /// sequence's track.
    pub(crate) fn note_forward(&self, seqs: &[usize]) {
        if let Ok(mut slot) = self.mtp.lock() {
            if let Some(mtp) = slot.as_mut() {
                mtp.forward = seqs.to_vec();
            }
        }
    }

    /// Feeds an attached head one forward's streams `x`, before the output mixer.
    pub(crate) fn observe(
        &self,
        ids: &Tensor,
        x: &Tensor,
        offsets: &[usize],
        context_lens: &[(usize, usize)],
    ) -> Result<()> {
        let mut slot = self
            .mtp
            .lock()
            .map_err(|_| hanzo_ml::Error::msg("zen6 MTP state poisoned"))?;
        match slot.as_mut() {
            Some(mtp) => mtp.observe(ids, x, offsets, context_lens),
            None => Ok(()),
        }
    }

    /// The streams of the rows the last forward took logits from, each row's position, and
    /// where the proposer takes the anchors picked among them. `None` without a head.
    pub(crate) fn last_spec(&self) -> Option<(Tensor, Vec<Vec<usize>>, AnchorPositions)> {
        let slot = self.mtp.lock().ok()?;
        let mtp = slot.as_ref()?;
        let (hidden, positions) = mtp.spec.clone()?;
        Some((hidden, positions, mtp.anchors.clone()))
    }

    /// Detach the head: forwards stop feeding it.
    pub(crate) fn detach_mtp(&self) {
        if let Ok(mut slot) = self.mtp.lock() {
            *slot = None;
        }
    }

    /// The embedding and output head, lent to a head that carries neither. The output head takes
    /// the streams' dtype, which is the embedding's.
    fn shared_heads(&self) -> SpeculativeSharedHeads {
        let embed = self.tok_embeddings.clone();
        let output = Arc::clone(&self.output);
        let dtype = self.tok_embeddings.embeddings().dtype();
        SpeculativeSharedHeads {
            embed: Arc::new(move |ids: &Tensor| embed.forward(ids)),
            lm_head: Arc::new(move |hidden: &Tensor| {
                output.forward(&hidden.to_dtype(dtype)?.contiguous()?)
            }),
        }
    }
}

/// Qwen3.8-Flash-Next is self-speculative: its head ships as a companion GGUF, the path `cfg`
/// names, and reads the target's streams. This impl is the one place that knowledge lives.
impl SelfSpeculative for ModelWeights {
    fn attach_mtp(&self, cfg: &MtpConfig) -> Result<Box<dyn SpeculativeProposer + Send + Sync>> {
        let path = cfg.resolve_path()?;
        let mut files = [std::fs::File::open(&path).map_err(|e| {
            hanzo_ml::Error::Msg(format!("failed to open MTP GGUF {}: {e}", path.display()))
        })?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        let file = Head::load(&mut ct, self.rotary.clone(), &self.device, self.dtype)?;

        // A head reads the target's streams and writes its vocabulary: both must match.
        let (vocab, hidden) = self.tok_embeddings.embeddings().dims2()?;
        if (file.hidden, file.head.streams) != (hidden, self.streams) {
            hanzo_ml::bail!(
                "the MTP head is {} wide over {} streams; the model is {hidden} wide over {}",
                file.hidden,
                file.head.streams,
                self.streams
            );
        }
        let embed_vocab = file.embed.as_ref().map(|e| e.dim(0)).transpose()?;
        for own in [embed_vocab, file.output.as_ref().map(|(_, v)| *v)]
            .into_iter()
            .flatten()
        {
            if own != vocab {
                hanzo_ml::bail!(
                    "the MTP head's vocabulary is {own} tokens; the model's is {vocab}"
                );
            }
        }

        let target = self.shared_heads();
        let heads = SpeculativeSharedHeads {
            embed: match file.embed {
                Some(table) => {
                    let embed = Embedding::new(table, hidden);
                    Arc::new(move |ids: &Tensor| embed.forward(ids))
                }
                None => target.embed,
            },
            lm_head: match file.output {
                Some((output, _)) => {
                    Arc::new(move |hidden: &Tensor| output.forward(&hidden.contiguous()?))
                }
                None => target.lm_head,
            },
        };
        let n_predict = cfg.n_predict.unwrap_or(default_n_predict(hidden)).max(1);
        let anchors = AnchorPositions::default();
        // A chain writes its rows past the last target position.
        let mtp = Mtp::new(
            file.head,
            heads.embed.clone(),
            anchors.clone(),
            self.max_seq_len + n_predict,
        );
        *self
            .mtp
            .lock()
            .map_err(|_| hanzo_ml::Error::msg("zen6 MTP state poisoned"))? = Some(mtp);
        Ok(Box::new(Qwen3_5MtpProposer::new(
            Box::new(Drafter {
                slot: self.mtp.clone(),
                device: self.device.clone(),
                dtype: self.tok_embeddings.embeddings().dtype(),
            }),
            heads,
            n_predict,
            anchors,
        )))
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use hanzo_ml::quantized::{gguf_file, GgmlDType, QTensor};
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand_isaac::Isaac64Rng;

    use super::*;
    use crate::models::hyper::reference as hc;
    use crate::models::quantized_zen6::tests::{gguf, model};
    use crate::models::qwen3_5_mtp::anchor_rows;
    use crate::sequence::test_sequence;
    use crate::speculative::{Lookup, SpeculativeKvCache, SpeculativeProposeBatchCtx};

    const SEQ: usize = 7;
    const TARGET_SEED: u64 = 11;
    /// A head that drafts at chance over this target, some of whose drafts it accepts.
    const HEAD_SEED: u64 = 26;

    /// What a head file carries of the token embedding and output projection.
    #[derive(Clone, Copy)]
    enum Tables {
        /// Neither: it borrows the target's, as `shared` files do.
        Borrowed,
        /// Copies of the target's.
        Copied,
        /// Tables of its own over a vocabulary of this many tokens.
        Own(usize),
    }

    fn read(bytes: &[u8]) -> Result<(gguf_file::Content, Cursor<Vec<u8>>)> {
        let mut cursor = Cursor::new(bytes.to_vec());
        let content = gguf_file::Content::read(&mut cursor)?;
        Ok((content, cursor))
    }

    fn meta(content: &gguf_file::Content, key: &str) -> usize {
        match content.metadata.get(&format!("{ARCH}.{key}")) {
            Some(gguf_file::Value::U32(v)) => *v as usize,
            other => panic!("{key}: {other:?}"),
        }
    }

    /// A head for the target GGUF `target`, laid out as the real files are: the target's metadata
    /// with the head counted in `block_count`, the target's attention block (`blk.3`) as the head's
    /// block, and the entry and exit shaped from the target's own tensors.
    fn head_gguf(target: &[u8], tables: Tables) -> Result<Vec<u8>> {
        let (content, mut cursor) = read(target)?;
        let dims = |name: &str| content.tensor_infos[name].shape.dims().to_vec();
        let depth = meta(&content, "block_count");
        let hidden = dims("token_embd.weight")[1];
        let block = format!("blk.{depth}");
        let mut shapes: Vec<(String, Vec<usize>)> = content
            .tensor_infos
            .iter()
            .filter_map(|(name, info)| {
                let rest = name.strip_prefix("blk.3.")?;
                Some((format!("{block}.{rest}"), info.shape.dims().to_vec()))
            })
            .collect();
        // The rng draws in a fixed order, whatever order the map yields the names in.
        shapes.sort();
        for part in ["norm", "down", "up"] {
            shapes.push((
                format!("{block}.nextn.hc_head_{part}.weight"),
                dims(&format!("output_hc_{part}.weight")),
            ));
        }
        shapes.push((
            format!("{block}.nextn.eh_proj.weight"),
            vec![hidden, 2 * hidden],
        ));
        shapes.push((format!("{block}.nextn.enorm.weight"), vec![hidden]));
        shapes.push((
            format!("{block}.nextn.hnorm.weight"),
            dims("output_hc_norm.weight"),
        ));
        if let Tables::Own(vocab) = tables {
            for name in ["token_embd.weight", "output.weight"] {
                shapes.push((name.to_string(), vec![vocab, hidden]));
            }
        }

        let mut rng = StdRng::seed_from_u64(HEAD_SEED);
        let mut owned = Vec::with_capacity(shapes.len() + 2);
        for (name, shape) in shapes {
            let n: usize = shape.iter().product();
            // Norm-like vectors sit near 1, as the target's do; the shared expert's gate does not.
            let (lo, hi) = if shape.len() == 1 && !name.contains("shexp") {
                (0.7, 1.3)
            } else {
                (-0.5, 0.5)
            };
            let data: Vec<f32> = (0..n).map(|_| rng.random_range(lo..hi)).collect();
            // The indexed MoE matmul runs on quantized banks only.
            let dtype = if name.contains("_exps.") {
                GgmlDType::Q8_0
            } else {
                GgmlDType::F32
            };
            let t = Tensor::from_vec(data, shape, &Device::Cpu)?;
            owned.push((name, QTensor::quantize(&t, dtype)?));
        }
        if let Tables::Copied = tables {
            for name in ["token_embd.weight", "output.weight"] {
                owned.push((
                    name.to_string(),
                    content.tensor(&mut cursor, name, &Device::Cpu)?,
                ));
            }
        }

        let mut metadata = content.metadata.clone();
        metadata.insert(
            format!("{ARCH}.block_count"),
            gguf_file::Value::U32(u32::try_from(depth + 1).expect("a tiny depth")),
        );
        metadata.insert(
            format!("{ARCH}.nextn_predict_layers"),
            gguf_file::Value::U32(1),
        );
        let kv: Vec<(&str, &gguf_file::Value)> =
            metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
        let named: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let mut file = Cursor::new(Vec::new());
        gguf_file::write(&mut file, &kv, &named)?;
        Ok(file.into_inner())
    }

    /// The head in `bytes` beside `target`, as `attach_mtp` builds it.
    fn beside(target: &ModelWeights, bytes: &[u8]) -> Result<Mtp> {
        let mut file = Cursor::new(bytes.to_vec());
        let mut readers = [&mut file];
        let mut ct = Content::from_readers(&mut readers)?;
        let file = Head::load(&mut ct, target.rotary.clone(), &Device::Cpu, DType::F32)?;
        Ok(Mtp::new(
            file.head,
            target.shared_heads().embed,
            AnchorPositions::default(),
            64,
        ))
    }

    /// The target in `target` with the head in `head` attached as the pipeline attaches it.
    fn attach(
        target: &[u8],
        head: &[u8],
        n_predict: usize,
        lookup: Option<Lookup>,
    ) -> Result<(ModelWeights, Box<dyn SpeculativeProposer + Send + Sync>)> {
        let model = model(target)?;
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let path = dir.path().join("mtp.gguf");
        std::fs::write(&path, head).map_err(hanzo_ml::Error::msg)?;
        let cfg = MtpConfig::new(path.to_string_lossy().into_owned(), Some(n_predict))
            .with_lookup(lookup);
        let proposer = model.mtp_proposer(&cfg)?;
        Ok((model, proposer))
    }

    fn f64s(t: &Tensor) -> Result<Vec<f64>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    }

    /// The largest deviation from `want` over its largest magnitude.
    fn relative(got: &[f64], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (g, w)| m.max((g - w).abs()));
        worst / scale
    }

    /// The head's shape, read from its file.
    struct Dims {
        depth: usize,
        hidden: usize,
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
        experts: usize,
        top_k: usize,
        eps: f64,
    }

    /// Every tensor in `bytes`, dequantized, row-major.
    fn weights(bytes: &[u8]) -> Result<(Dims, HashMap<String, Vec<f64>>)> {
        let (content, mut cursor) = read(bytes)?;
        let eps = match content.metadata[&format!("{ARCH}.attention.layer_norm_rms_epsilon")] {
            gguf_file::Value::F32(v) => f64::from(v),
            ref other => panic!("eps: {other:?}"),
        };
        let dims = Dims {
            depth: meta(&content, "block_count") - 1,
            hidden: meta(&content, "embedding_length"),
            heads: meta(&content, "attention.head_count"),
            kv_heads: meta(&content, "attention.head_count_kv"),
            head_dim: meta(&content, "attention.key_length"),
            experts: meta(&content, "expert_count"),
            top_k: meta(&content, "expert_used_count"),
            eps,
        };
        let names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        let mut out = HashMap::new();
        for name in names {
            let t = content
                .tensor(&mut cursor, &name, &Device::Cpu)?
                .dequantize(&Device::Cpu)?;
            out.insert(name, f64s(&t)?);
        }
        Ok((dims, out))
    }

    /// One head step over an empty history, in f64, by the module's graph: the draft's
    /// streams and logits for token `tok` over target streams `hs`. The target lends `table` its
    /// embedding and output projection. `swap` puts the entry's halves the wrong way round.
    fn reference(
        d: &Dims,
        w: &HashMap<String, Vec<f64>>,
        table: &HashMap<String, Vec<f64>>,
        tok: usize,
        hs: &[f64],
        swap: bool,
    ) -> (Vec<f64>, Vec<f64>) {
        let b = |name: &str| &w[&format!("blk.{}.{name}.weight", d.depth)][..];
        let (h, eps) = (d.hidden, d.eps);
        let silu = |x: f64| x * hc::sigmoid(x);
        let branch = |x: &[f64], prefix: &str| {
            let xn = hc::norm(x, b(&format!("{prefix}_norm")), h, eps);
            let inj = hc::linear(b(&format!("{prefix}_inject")), &xn);
            let u = hc::mix(
                &xn,
                b(&format!("{prefix}_down")),
                b(&format!("{prefix}_up")),
                h,
            );
            (inj, u)
        };

        let e = &table["token_embd.weight"][tok * h..(tok + 1) * h];
        let en = hc::norm(e, b("nextn.enorm"), h, eps);
        let hn = hc::norm(hs, b("nextn.hnorm"), h, eps);
        let r: Vec<f64> = hn
            .chunks(h)
            .flat_map(|hs| {
                let cat = if swap {
                    [hs, &en[..]].concat()
                } else {
                    [&en[..], hs].concat()
                };
                hc::linear(b("nextn.eh_proj"), &cat)
            })
            .collect();

        // Attention over the one pair the history holds: softmax over a single key is 1, so each
        // head reads its KV head's value, gated by its half of the query projection.
        let (inj, u) = branch(&r, "hc_attn");
        let q_gate = hc::linear(b("attn_q"), &u);
        let v = hc::linear(b("attn_v"), &u);
        let (hd, groups) = (d.head_dim, d.heads / d.kv_heads);
        let y: Vec<f64> = (0..d.heads)
            .flat_map(|i| {
                let gate = &q_gate[(2 * i + 1) * hd..(2 * i + 2) * hd];
                let v = &v[(i / groups) * hd..(i / groups + 1) * hd];
                v.iter()
                    .zip(gate)
                    .map(|(v, g)| v * hc::sigmoid(*g))
                    .collect::<Vec<_>>()
            })
            .collect();
        let r = hc::combine(&r, &hc::linear(b("attn_output"), &y), &inj);

        // Softmax routing to the top k experts, renormalized, beside the sigmoid-gated shared one.
        let (inj, u) = branch(&r, "hc_ffn");
        let logits = hc::linear(b("ffn_gate_inp"), &u);
        let max = logits.iter().fold(f64::MIN, |m, v| m.max(*v));
        let p: Vec<f64> = logits.iter().map(|v| (v - max).exp()).collect();
        let mut order: Vec<usize> = (0..d.experts).collect();
        order.sort_by(|a, c| p[*c].total_cmp(&p[*a]));
        let top = &order[..d.top_k];
        let total: f64 = top.iter().map(|e| p[*e]).sum();
        let swiglu = |gate: &[f64], up: &[f64], down: &[f64]| {
            let a: Vec<f64> = hc::linear(gate, &u)
                .iter()
                .zip(hc::linear(up, &u))
                .map(|(g, up)| silu(*g) * up)
                .collect();
            hc::linear(down, &a)
        };
        let expert = |bank: &str, e: usize| {
            let all = b(bank);
            let per = all.len() / d.experts;
            &all[e * per..(e + 1) * per]
        };
        let mut moe = vec![0f64; h];
        for &e in top {
            let y = swiglu(
                expert("ffn_gate_exps", e),
                expert("ffn_up_exps", e),
                expert("ffn_down_exps", e),
            );
            for (m, y) in moe.iter_mut().zip(y) {
                *m += p[e] / total * y;
            }
        }
        let shared = swiglu(b("ffn_gate_shexp"), b("ffn_up_shexp"), b("ffn_down_shexp"));
        let gate = hc::sigmoid(hc::linear(b("ffn_gate_inp_shexp"), &u)[0]);
        for (m, y) in moe.iter_mut().zip(shared) {
            *m += gate * y;
        }
        let r = hc::combine(&r, &moe, &inj);

        let mixed = hc::mix(
            &hc::norm(&r, b("nextn.hc_head_norm"), h, eps),
            b("nextn.hc_head_down"),
            b("nextn.hc_head_up"),
            h,
        );
        let logits = hc::linear(&table["output.weight"], &mixed);
        (r, logits)
    }

    fn positions(at: u32) -> Result<Tensor> {
        Tensor::new(&[[[at]], [[at]], [[at]]], &Device::Cpu)
    }

    /// One draft step of `mtp` for sequence `seq` at `at`: its streams and logits.
    fn draft(
        mtp: &mut Mtp,
        heads: &SpeculativeSharedHeads,
        seq: usize,
        tok: u32,
        hidden: &Tensor,
        at: u32,
    ) -> Result<(Tensor, Tensor)> {
        let embeds = (heads.embed)(&Tensor::new(&[[tok]], &Device::Cpu)?)?;
        let out = mtp.draft(&[seq], &embeds, hidden, &positions(at)?)?;
        Ok((out.carry, (heads.lm_head)(&out.head)?))
    }

    /// The first step of a sequence matches the head's graph computed in f64. The routed experts
    /// sit in Q8_0 banks, whose CPU matmul quantizes its input to Q8_0 as well, so the step lands
    /// within that rounding rather than at f32 precision; a head with its entry's halves swapped
    /// lands far outside it.
    #[test]
    fn a_first_step_matches_the_reference() -> Result<()> {
        const BOUND: f64 = 5e-3;
        let target_bytes = gguf(TARGET_SEED)?;
        let head_bytes = head_gguf(&target_bytes, Tables::Borrowed)?;
        let target = model(&target_bytes)?;
        let heads = target.shared_heads();
        let mut mtp = beside(&target, &head_bytes)?;

        let (d, w) = weights(&head_bytes)?;
        let (_, table) = weights(&target_bytes)?;
        let mut rng = StdRng::seed_from_u64(3);
        let hs: Vec<f32> = (0..target.streams * d.hidden)
            .map(|_| rng.random_range(-1.0..1.0))
            .collect();
        let hidden = Tensor::from_vec(hs.clone(), (1, 1, hs.len()), &Device::Cpu)?;
        let hs: Vec<f64> = hs.into_iter().map(f64::from).collect();
        let (carry, logits) = draft(&mut mtp, &heads, SEQ, 5, &hidden, 9)?;

        let (want_carry, want_logits) = reference(&d, &w, &table, 5, &hs, false);
        let (carry, logits) = (f64s(&carry)?, f64s(&logits)?);
        let (carry_err, logits_err) = (
            relative(&carry, &want_carry),
            relative(&logits, &want_logits),
        );
        assert!(
            carry_err < BOUND,
            "streams are {carry_err:.2e} off the reference"
        );
        assert!(
            logits_err < BOUND,
            "logits are {logits_err:.2e} off the reference"
        );

        let (_, swapped) = reference(&d, &w, &table, 5, &hs, true);
        let control = relative(&logits, &swapped);
        assert!(
            control > 10.0 * BOUND,
            "a swapped entry is only {control:.2e} off"
        );
        Ok(())
    }

    /// A draft over a KV built forward by forward, through a chunked prefill, a decode, a verify
    /// whose last draft was rejected and a chained step, equals a draft over the same pairs
    /// written in one forward; the same draft with no history differs.
    #[test]
    fn a_draft_reads_the_whole_sequence_however_it_was_written() -> Result<()> {
        const LEN: usize = 16;
        let target_bytes = gguf(TARGET_SEED)?;
        let head_bytes = head_gguf(&target_bytes, Tables::Borrowed)?;
        let target = model(&target_bytes)?;
        let heads = target.shared_heads();
        let (n, h) = (target.streams, target.tok_embeddings.embeddings().dim(1)?);

        let mut rng = StdRng::seed_from_u64(17);
        let toks: Vec<u32> = (0..LEN).map(|_| rng.random_range(0..24)).collect();
        let mut rows = |count: usize| -> Result<Tensor> {
            let data: Vec<f32> = (0..count * n * h)
                .map(|_| rng.random_range(-1.0..1.0))
                .collect();
            Tensor::from_vec(data, (count, n, h), &Device::Cpu)
        };
        let streams = rows(LEN)?;
        let rejected = rows(2)?;

        // `ids` and their streams `x` `[rows, n, h]` as one forward from `at`.
        let feed = |mtp: &mut Mtp, ids: &[u32], x: &Tensor, at: usize| -> Result<()> {
            mtp.forward = vec![SEQ];
            let len = ids.len();
            mtp.observe(
                &Tensor::new(ids, &Device::Cpu)?.unsqueeze(0)?,
                &x.unsqueeze(0)?,
                &[at],
                &[(0, len)],
            )
        };
        let at = |p: usize| -> Result<Tensor> { streams.narrow(0, p, 1)?.reshape((1, 1, n * h)) };
        let logits = |d: (Tensor, Tensor)| f64s(&d.1);

        let mut mtp = beside(&target, &head_bytes)?;
        feed(&mut mtp, &toks[..5], &streams.narrow(0, 0, 5)?, 0)?;
        feed(&mut mtp, &toks[5..9], &streams.narrow(0, 5, 4)?, 5)?;
        feed(&mut mtp, &toks[9..10], &streams.narrow(0, 9, 1)?, 9)?;
        // Anchor 10 and drafts 11, 12, 13: 11 is accepted, 12 is not, and the verifier emits 12.
        let verify = Tensor::cat(&[streams.narrow(0, 10, 2)?, rejected], 0)?;
        feed(&mut mtp, &[toks[10], toks[11], 20, 21], &verify, 10)?;
        let first = draft(&mut mtp, &heads, SEQ, toks[12], &at(11)?, 11)?;
        // A chained step writes a row the next forward must replace.
        draft(&mut mtp, &heads, SEQ, 3, &first.0, 12)?;
        let first = logits(first)?;
        feed(&mut mtp, &toks[12..14], &streams.narrow(0, 12, 2)?, 12)?;
        let second = logits(draft(&mut mtp, &heads, SEQ, toks[14], &at(13)?, 13)?)?;

        for (got, upto) in [(&first, 11), (&second, 13)] {
            let mut once = beside(&target, &head_bytes)?;
            feed(
                &mut once,
                &toks[..=upto],
                &streams.narrow(0, 0, upto + 1)?,
                0,
            )?;
            let want = logits(draft(
                &mut once,
                &heads,
                SEQ,
                toks[upto + 1],
                &at(upto)?,
                u32::try_from(upto).expect("a tiny position"),
            )?)?;
            let err = relative(got, &want);
            assert!(err < 1e-6, "a draft at {upto} is {err:.2e} off one pass");
        }

        let alone = logits(draft(&mut mtp, &heads, SEQ + 1, toks[14], &at(13)?, 13)?)?;
        let err = relative(&alone, &second);
        assert!(
            err > 1e-3,
            "a draft with no history is only {err:.2e} off one with it"
        );
        Ok(())
    }

    /// A self-contained head drafts through its own embedding and output projection: holding
    /// copies of the target's, it drafts exactly as a shared head; holding its own, differently.
    #[test]
    fn a_self_contained_head_uses_its_own_tables() -> Result<()> {
        let target_bytes = gguf(TARGET_SEED)?;
        let vocab = model(&target_bytes)?.tok_embeddings.embeddings().dim(0)?;
        let first_draft = |tables: Tables| -> Result<Vec<f64>> {
            let head_bytes = head_gguf(&target_bytes, tables)?;
            let (model, mut proposer) = attach(&target_bytes, &head_bytes, 1, None)?;
            let prompt = [1u32, 5, 3, 7];
            let logits = forward(&model, &prompt, 0, prompt.len(), false)?;
            let anchor = argmax(&logits.get(0)?)?;
            let mut toks = prompt.to_vec();
            toks.push(anchor);
            let proposal = propose(&model, &mut *proposer, &toks, 0)?;
            f64s(
                proposal
                    .logits
                    .as_ref()
                    .expect("an MTP draft carries its logits"),
            )
        };
        let borrowed = first_draft(Tables::Borrowed)?;
        assert_eq!(first_draft(Tables::Copied)?, borrowed);
        let own = first_draft(Tables::Own(vocab))?;
        assert!(
            relative(&own, &borrowed) > 1e-2,
            "its own tables went unused"
        );
        Ok(())
    }

    #[test]
    fn a_head_that_does_not_fit_is_refused() -> Result<()> {
        let target_bytes = gguf(TARGET_SEED)?;
        let refusal = |head: &[u8]| match attach(&target_bytes, head, 1, None) {
            Ok(_) => panic!("attached"),
            Err(e) => e.to_string(),
        };
        // The target's own file counts no head.
        let err = refusal(&target_bytes);
        assert!(err.contains("carries no MTP head"), "{err}");
        let vocab = model(&target_bytes)?.tok_embeddings.embeddings().dim(0)?;
        let err = refusal(&head_gguf(&target_bytes, Tables::Own(vocab + 8))?);
        assert!(err.contains("vocabulary"), "{err}");
        Ok(())
    }

    /// `toks[at..at + len]` through `model` as one forward: the logits of every row for a verify,
    /// else of the last, as `[rows, vocab]`.
    fn forward(
        model: &ModelWeights,
        toks: &[u32],
        at: usize,
        len: usize,
        verify: bool,
    ) -> Result<Tensor> {
        model.note_forward(&[SEQ]);
        model.cache.hybrid().expect_verify(verify.then_some(len));
        let ids = Tensor::new(&toks[at..at + len], &Device::Cpu)?.unsqueeze(0)?;
        let prior = vec![toks[at.saturating_sub(2)..at].to_vec()];
        let rows = if verify { (0, len) } else { (len - 1, 1) };
        model
            .forward(&ids, &prior, &[at], vec![rows], None)?
            .squeeze(0)
    }

    fn argmax(row: &Tensor) -> Result<u32> {
        row.argmax(D::Minus1)?.to_scalar::<u32>()
    }

    /// The draft for `toks`, whose last token was sampled from logits row `row` of the last
    /// forward, handed over as the speculative driver hands it.
    fn propose(
        model: &ModelWeights,
        proposer: &mut (dyn SpeculativeProposer + Send + Sync),
        toks: &[u32],
        row: usize,
    ) -> Result<crate::speculative::SpeculativeProposal> {
        let (hidden, positions, anchors) = model
            .last_spec()
            .expect("an attached head sees every forward");
        let hidden = anchor_rows(&hidden, &positions, &[(0, row)], &anchors)?;
        let seq = test_sequence(toks.to_vec(), None);
        let anchor = toks[toks.len() - 1];
        let mut batch = proposer.propose(
            SpeculativeProposeBatchCtx {
                sampled_tokens: &[anchor],
                sampled_tokens_emitted: true,
                seq_ids: &[SEQ],
                base_lens: &[toks.len() - 1],
                sequences: &[&seq],
                cache: SpeculativeKvCache::Normal,
                target_hiddens: Some(hidden),
                target_hidden_layers: None,
                rng: Arc::new(Mutex::new(Isaac64Rng::seed_from_u64(0))),
            },
            None,
        )?;
        Ok(batch.proposals.remove(0))
    }

    /// `steps` greedy tokens after `prompt`, one forward each, and the gap between the best and
    /// second-best logit behind each.
    fn serial(target: &[u8], prompt: &[u32], steps: usize) -> Result<(Vec<u32>, Vec<f32>)> {
        let model = model(target)?;
        let mut toks = prompt.to_vec();
        let mut margins = Vec::with_capacity(steps);
        let mut logits = forward(&model, &toks, 0, toks.len(), false)?;
        for _ in 0..steps {
            let row = logits.get(0)?;
            let mut sorted = row.to_vec1::<f32>()?;
            sorted.sort_by(|a, b| b.total_cmp(a));
            margins.push(sorted[0] - sorted[1]);
            toks.push(argmax(&row)?);
            logits = forward(&model, &toks, toks.len() - 1, 1, false)?;
        }
        Ok((toks[prompt.len()..].to_vec(), margins))
    }

    /// `steps` greedy tokens after `prompt` with the head drafting: each draft verified in one
    /// forward, the longest matching prefix kept, and the rejected rows rolled back as the
    /// speculative driver does. Also the drafted and accepted counts.
    fn speculative(
        target: &[u8],
        head: &[u8],
        prompt: &[u32],
        steps: usize,
        n_predict: usize,
        lookup: Option<Lookup>,
    ) -> Result<(Vec<u32>, usize, usize)> {
        let (model, mut proposer) = attach(target, head, n_predict, lookup)?;
        let slot = model.cache.hybrid().state_indices_host().expect("a slot")[0] as usize;
        let mut toks = prompt.to_vec();
        let logits = forward(&model, &toks, 0, toks.len(), false)?;
        toks.push(argmax(&logits.get(0)?)?);
        let (mut row, mut drafted, mut accepted) = (0, 0, 0);
        while toks.len() < prompt.len() + steps {
            let drafts = propose(&model, &mut *proposer, &toks, row)?.tokens;
            let at = toks.len() - 1;
            let mut verify = toks.clone();
            verify.extend(&drafts);
            let logits = forward(&model, &verify, at, drafts.len() + 1, true)?;
            let mut kept = 0;
            while kept < drafts.len() && argmax(&logits.get(kept)?)? == drafts[kept] {
                kept += 1;
            }
            toks.extend(&drafts[..kept]);
            toks.push(argmax(&logits.get(kept)?)?);
            let rejected = drafts.len() - kept;
            if rejected > 0 {
                // As the engine rewinds an eager cache: every K/V, the QSA index's included.
                let mut cache = model.cache.hybrid();
                cache.truncate_attention_to(at + 1 + kept)?;
                cache.rewind_recurrent(slot, rejected)?;
            }
            row = kept;
            drafted += drafts.len();
            accepted += kept;
        }
        proposer.retain_seqs(&[]);
        let released = model
            .mtp
            .lock()
            .expect("MTP state")
            .as_ref()
            .is_some_and(|mtp| mtp.tracks.is_empty());
        assert!(released, "a finished sequence kept its track");
        toks.truncate(prompt.len() + steps);
        Ok((toks[prompt.len()..].to_vec(), drafted, accepted))
    }

    /// The target's logits are the same bits with a head attached as without.
    #[test]
    fn an_attached_head_leaves_the_target_untouched() -> Result<()> {
        let target_bytes = gguf(TARGET_SEED)?;
        let head_bytes = head_gguf(&target_bytes, Tables::Borrowed)?;
        let plain = model(&target_bytes)?;
        let (with, _proposer) = attach(&target_bytes, &head_bytes, 1, None)?;
        let toks = [1u32, 5, 9, 3, 7, 2, 11, 4];
        for (at, len) in [(0, 5), (5, 1), (6, 2)] {
            let bits = |m: &ModelWeights| -> Result<Vec<u32>> {
                Ok(forward(m, &toks, at, len, len > 1 && at > 0)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(bits(&with)?, bits(&plain)?, "rows {at}..{}", at + len);
        }
        Ok(())
    }

    /// Greedy decoding with the head drafting, alone and continued by prompt lookup, emits the
    /// tokens serial decoding does. A verify runs its rows as one chunk while serial decoding runs
    /// them one at a time, and the two round differently, so the fixture's serial steps must each
    /// win by more than that rounding for equality to be the claim.
    #[test]
    fn greedy_speculative_decoding_matches_serial() -> Result<()> {
        const STEPS: usize = 20;
        let target_bytes = gguf(TARGET_SEED)?;
        let head_bytes = head_gguf(&target_bytes, Tables::Borrowed)?;
        let prompt = [1u32, 5, 3, 7, 2, 1, 5, 3];
        let (serial, margins) = serial(&target_bytes, &prompt, STEPS)?;
        let closest = margins.iter().fold(f32::MAX, |m, v| m.min(*v));
        assert!(closest > 1e-2, "a serial step wins by only {closest:e}");

        let mut total_accepted = 0;
        for (n_predict, lookup) in [(1, None), (2, None), (1, Some(Lookup::new(2, 2, 3)?))] {
            let (spec, drafted, accepted) = speculative(
                &target_bytes,
                &head_bytes,
                &prompt,
                STEPS,
                n_predict,
                lookup,
            )?;
            assert_eq!(spec, serial, "n_predict {n_predict}, lookup {lookup:?}");
            assert!(
                accepted < drafted,
                "every draft was accepted: no rollback ran"
            );
            total_accepted += accepted;
        }
        assert!(total_accepted > 0, "no draft was ever accepted");
        Ok(())
    }
}
