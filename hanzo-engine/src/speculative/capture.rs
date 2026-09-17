//! Target-side capture of the decoder-layer hidden states a parallel-block draft
//! (DSpark, DFlash) fuses into its context.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use hanzo_ml::{Result, Tensor};

/// The confirmed-prefix hidden states of ONE sequence: one `[prefix_len, hidden]` tensor per
/// captured layer, in the HF `hidden_states[i + 1]` convention (the output of layer `i`).
///
/// A model composes one of these and drives it from its layer loop. It is off by default,
/// and then `layers_for` returns nothing, so the loop is byte-identical to the
/// non-speculative path.
///
/// Each captured forward truncates the buffer to its start position before appending. That
/// seeds the prompt on prefill (start 0), extends it on each decode, and drops rejected drafts
/// on the next verify forward. Positions alone cannot tell that rollback from a *different*
/// sequence running at a smaller offset, so the buffer is keyed by sequence id: a change of
/// owner reseeds it instead of splicing two sequences' rows together, and a multi-sequence
/// forward — which is never captured — drops it.
#[derive(Default)]
pub struct HiddenPrefixCapture {
    enabled: AtomicBool,
    layers: Mutex<Vec<usize>>,
    state: Mutex<CaptureState>,
}

#[derive(Default)]
struct CaptureState {
    prefix: Vec<Tensor>,
    /// The sequence `prefix` describes.
    owner: Option<usize>,
    /// The single sequence the next forward runs; `None` when it runs several.
    forward: Option<usize>,
}

impl CaptureState {
    fn drop_prefix(&mut self) {
        self.prefix.clear();
        self.owner = None;
    }
}

impl HiddenPrefixCapture {
    /// Capture `layers` (a draft checkpoint's `target_layer_ids`); an empty list turns capture
    /// off. Either way the buffer is dropped, so no sequence inherits another's rows.
    pub fn set_layers(&self, layers: Vec<usize>) {
        self.enabled.store(!layers.is_empty(), Ordering::Relaxed);
        if let Ok(mut current) = self.layers.lock() {
            *current = layers;
        }
        if let Ok(mut state) = self.state.lock() {
            state.drop_prefix();
        }
    }

    /// Names the sequences the next forward runs.
    pub fn note_forward(&self, seq_ids: &[usize]) {
        if let Ok(mut state) = self.state.lock() {
            state.forward = match seq_ids {
                [id] => Some(*id),
                _ => None,
            };
        }
    }

    /// The layers to snapshot during a forward over `num_seqs` sequences: none while capture
    /// is off, and none for a multi-sequence forward. After one of those the buffer describes
    /// no sequence's confirmed prefix, so it is dropped and the proposer stands down for the
    /// survivors rather than drafting from rows that stopped tracking them.
    pub fn layers_for(&self, num_seqs: usize) -> Vec<usize> {
        if !self.enabled.load(Ordering::Relaxed) {
            return Vec::new();
        }
        if num_seqs == 1 {
            return self
                .layers
                .lock()
                .map(|layers| layers.clone())
                .unwrap_or_default();
        }
        if let Ok(mut state) = self.state.lock() {
            state.drop_prefix();
        }
        Vec::new()
    }

    /// Fold one forward's captured rows into the buffer. `start_pos` is the KV position of the
    /// forward's first token; `this_forward[k]` is the k-th captured layer, shaped
    /// `[1, query_len, hidden]` or `[query_len, hidden]`.
    pub fn fold(&self, start_pos: usize, this_forward: Vec<Tensor>) -> Result<()> {
        let Ok(mut state) = self.state.lock() else {
            return Ok(());
        };
        if state.owner != state.forward || state.prefix.len() != this_forward.len() {
            state.drop_prefix();
        }
        state.owner = state.forward;

        let mut merged = Vec::with_capacity(this_forward.len());
        for (k, cur) in this_forward.into_iter().enumerate() {
            let rows = match cur.dims() {
                [1, _, _] => cur.squeeze(0)?,
                [_, _] => cur,
                other => hanzo_ml::bail!("unexpected hidden capture shape {other:?}"),
            };
            let row = match state.prefix.get(k) {
                // Continuous capture: truncate to the forward's start and append the new rows.
                Some(prev) if start_pos <= prev.dim(0)? => {
                    if start_pos == 0 {
                        rows
                    } else {
                        Tensor::cat(&[&prev.narrow(0, 0, start_pos)?, &rows], 0)?
                    }
                }
                // A fresh prefill, or a gap the buffer cannot bridge (a prefix-cache hit starts
                // past 0 with nothing captured before it). The buffer is then shorter than the
                // sequence, which is what tells the proposer to stand down.
                _ => rows,
            };
            merged.push(row);
        }
        state.prefix = merged;
        Ok(())
    }

    /// The buffer, in capture-layer order. Clones share storage, so this is cheap.
    pub fn hiddens(&self) -> Vec<Tensor> {
        self.state
            .lock()
            .map(|state| state.prefix.clone())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    /// `rows` rows of width 2 whose first column is `tag`, so a row's origin stays readable
    /// after the buffer has been truncated and re-appended.
    fn rows(tag: f32, rows: usize) -> Result<Tensor> {
        let data: Vec<f32> = (0..rows).flat_map(|r| [tag, r as f32]).collect();
        Tensor::from_vec(data, (rows, 2), &Device::Cpu)
    }

    fn tags(capture: &HiddenPrefixCapture) -> Result<Vec<f32>> {
        match capture.hiddens().first() {
            Some(t) => Ok(t.to_vec2::<f32>()?.into_iter().map(|r| r[0]).collect()),
            None => Ok(Vec::new()),
        }
    }

    #[test]
    fn off_by_default_and_captures_nothing() {
        let capture = HiddenPrefixCapture::default();
        assert!(capture.layers_for(1).is_empty());
        assert!(capture.hiddens().is_empty());
    }

    #[test]
    fn one_sequence_seeds_extends_and_rolls_back() -> Result<()> {
        let capture = HiddenPrefixCapture::default();
        capture.set_layers(vec![3]);
        capture.note_forward(&[7]);
        assert_eq!(capture.layers_for(1), vec![3]);

        capture.fold(0, vec![rows(1.0, 4)?])?; // prefill
        capture.fold(4, vec![rows(2.0, 3)?])?; // verify forward: anchor + two drafts
        assert_eq!(tags(&capture)?, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        // One draft was rejected: the next forward starts at 6 and drops the stale row.
        capture.fold(6, vec![rows(3.0, 1)?])?;
        assert_eq!(tags(&capture)?, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0]);
        Ok(())
    }

    /// A is captured, then B runs at a smaller offset. By position alone that is a rollback, and
    /// B would draft from A's hidden states for the rest of its life. The owner check reseeds
    /// instead, leaving a buffer shorter than B's position — the proposer's cue to stand down.
    #[test]
    fn another_sequence_never_inherits_the_buffer() -> Result<()> {
        let capture = HiddenPrefixCapture::default();
        capture.set_layers(vec![3]);

        capture.note_forward(&[1]);
        capture.fold(0, vec![rows(1.0, 1000)?])?;

        capture.note_forward(&[2]);
        capture.fold(500, vec![rows(2.0, 1)?])?;
        assert_eq!(tags(&capture)?, vec![2.0], "B spliced onto A's rows");

        // B's own prefill from 0 is a legitimate seed.
        capture.fold(0, vec![rows(2.0, 8)?])?;
        assert_eq!(tags(&capture)?.len(), 8);
        Ok(())
    }

    #[test]
    fn a_multi_sequence_forward_drops_the_buffer() -> Result<()> {
        let capture = HiddenPrefixCapture::default();
        capture.set_layers(vec![3]);
        capture.note_forward(&[1]);
        capture.fold(0, vec![rows(1.0, 16)?])?;

        capture.note_forward(&[1, 2]);
        assert!(capture.layers_for(2).is_empty());
        assert!(capture.hiddens().is_empty(), "a frozen buffer outlived a batched forward");

        // The survivor resumes alone mid-sequence: nothing to extend, so it stays short.
        capture.note_forward(&[1]);
        capture.fold(17, vec![rows(1.0, 1)?])?;
        assert_eq!(tags(&capture)?.len(), 1);
        Ok(())
    }

    #[test]
    fn a_leading_batch_dim_of_one_is_squeezed() -> Result<()> {
        let capture = HiddenPrefixCapture::default();
        capture.set_layers(vec![0]);
        capture.note_forward(&[1]);
        capture.fold(0, vec![rows(1.0, 5)?.unsqueeze(0)?])?;
        assert_eq!(capture.hiddens()[0].dims(), &[5, 2]);
        Ok(())
    }
}
