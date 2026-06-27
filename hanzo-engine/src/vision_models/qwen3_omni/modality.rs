#![allow(dead_code)]

//! Extensible modality fusion for Qwen3-Omni — ONE generic fusion path.
//!
//! A *modality* is anything that turns a raw, pre-tensorized payload (mel / pixels / frames) into
//! Thinker-space token embeddings, identified by the placeholder token id it replaces in the text
//! stream. [`fuse_modalities`] is the single, modality-agnostic scatter that every modality flows
//! through; to add a NEW native modality you implement [`ModalityEncoder`] and register the encoder
//! — the thinker, the talker, and this function never change. (Values, not places: a modality is
//! identified by the *value* of its placeholder token id, not by a hard-wired branch.)

use std::collections::HashMap;

use hanzo_ml::{DType, Device, Result, Tensor};

/// Raw, pre-tensorized modality payload handed to an encoder (already mel/pixels/frames, not bytes).
pub enum ModalityInput {
    /// Log-mel features `[1, num_mel_bins, T]`.
    Audio(Tensor),
    /// Pre-patchified image pixels `[num_patches, in_chans*temporal_patch*patch*patch]` together with
    /// the `[num_images, 3]` (t, h, w) patch grid the vision tower lays them out on. The grid is
    /// explicit (not square-derived) so non-square / multi-frame images fuse correctly.
    Image { pixels: Tensor, grid_thw: Tensor },
    /// Pre-patchified video pixels + `[num_videos, 3]` (t, h, w) patch grid (same tower as `Image`).
    Video { pixels: Tensor, grid_thw: Tensor },
}

/// A modality maps a raw [`ModalityInput`] into Thinker-space token embeddings and declares the
/// placeholder token id whose positions it fills. Implement this and register the encoder to add a
/// new native modality with ZERO changes to the thinker/talker/fusion.
pub trait ModalityEncoder: Send + Sync {
    /// The text-stream token id whose positions this encoder's output is scattered into.
    fn placeholder_token(&self) -> u32;
    /// Encode one payload into `[n_tokens, thinker_hidden]` Thinker-space token embeddings.
    fn encode(&self, input: &ModalityInput, device: &Device) -> Result<Tensor>;
}

/// THE single generic fusion: replace the rows of `embeds` whose `input_ids` equal an encoder's
/// placeholder token with that encoder's output, in sequence order.
///
/// Each `(placeholder, input)` in `inputs` is processed left-to-right: the encoder registered for
/// `placeholder` runs on `input` to produce `[n, H]`, and those `n` rows are written into the next
/// `n` still-unfilled positions where `input_ids == placeholder`. Multiple inputs that share one
/// placeholder consume consecutive runs of those positions, so several clips of the same modality
/// compose naturally. With no inputs the embeddings are returned unchanged.
///
/// The write is a scatter expressed as an `index_add` of the delta `(new_rows − current_rows)` at the
/// target row indices — device-agnostic, and exact because each placeholder position is unique
/// (mirrors the established `inputs_merger` idiom used by the other multimodal models in this crate).
pub fn fuse_modalities(
    embeds: &Tensor,    // [1, T, H]
    input_ids: &Tensor, // [1, T]
    encoders: &[Box<dyn ModalityEncoder>],
    inputs: &[(u32, ModalityInput)],
    device: &Device,
) -> Result<Tensor> {
    if inputs.is_empty() {
        return Ok(embeds.clone());
    }

    let (b, t, h) = embeds.dims3()?;
    let mut flat = embeds.reshape((b * t, h))?;
    let ids: Vec<u32> = input_ids
        .flatten_all()?
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;

    // Per-placeholder cursor: how many positions of that token id have already been filled, so that
    // repeated inputs of the same modality land in consecutive (non-overlapping) runs.
    let mut consumed: HashMap<u32, usize> = HashMap::new();

    for (placeholder, input) in inputs {
        let encoder = encoders
            .iter()
            .find(|e| e.placeholder_token() == *placeholder)
            .ok_or_else(|| {
                hanzo_ml::Error::msg(format!(
                    "fuse_modalities: no registered encoder for placeholder token {placeholder}"
                ))
            })?;

        let feats = encoder.encode(input, device)?;
        let (n, fh) = feats.dims2()?;
        if fh != h {
            hanzo_ml::bail!(
                "fuse_modalities: encoder for {placeholder} produced width {fh}, expected {h}"
            );
        }

        let start = *consumed.get(placeholder).unwrap_or(&0);
        let positions: Vec<u32> = ids
            .iter()
            .enumerate()
            .filter(|(_, &id)| id == *placeholder)
            .map(|(i, _)| i as u32)
            .skip(start)
            .take(n)
            .collect();
        if positions.len() != n {
            hanzo_ml::bail!(
                "fuse_modalities: {n} embeddings for placeholder {placeholder} but only {} free \
                 placeholder position(s) remain in the sequence",
                positions.len()
            );
        }
        *consumed.entry(*placeholder).or_insert(0) += n;

        // Row scatter: flat[positions] := feats. index_add of the delta leaves all other rows intact.
        let idx = Tensor::new(positions.as_slice(), device)?;
        let current = flat.index_select(&idx, 0)?;
        let delta = (feats.to_dtype(flat.dtype())? - current)?;
        flat = flat.index_add(&idx, &delta, 0)?;
    }

    flat.reshape((b, t, h))
}
