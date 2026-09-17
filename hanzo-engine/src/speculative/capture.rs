//! Target-side capture of the decoder-layer hidden states a parallel-block draft
//! (DSpark, DFlash) fuses into its context.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Mutex,
};

use hanzo_ml::{Result, Tensor};

/// Captured hiddens of one sequence: every layer holds positions `start..end()`, one
/// `[rows, hidden]` tensor each, in the HF `hidden_states[i + 1]` convention (the output of
/// layer `i`).
#[derive(Clone)]
pub struct HiddenWindow {
    pub start: usize,
    pub layers: Vec<Tensor>,
}

impl HiddenWindow {
    /// `layers` cover `start..`; a leading batch dim of 1 is squeezed.
    pub fn new(start: usize, layers: Vec<Tensor>) -> Result<Self> {
        let layers = layers
            .into_iter()
            .map(|t| match t.dims() {
                [1, _, _] => t.squeeze(0),
                [_, _] => Ok(t),
                other => hanzo_ml::bail!("unexpected hidden capture shape {other:?}"),
            })
            .collect::<Result<_>>()?;
        Ok(Self { start, layers })
    }

    /// One past the last position held.
    pub fn end(&self) -> Result<usize> {
        let rows = self.layers.first().map(|t| t.dim(0)).transpose()?;
        Ok(self.start + rows.unwrap_or(0))
    }

    /// Every layer's rows for positions `from..to`, which must lie inside the window.
    pub fn rows(&self, from: usize, to: usize) -> Result<Vec<Tensor>> {
        if from < self.start || to < from || to > self.end()? {
            hanzo_ml::bail!(
                "hidden window holds {}..{}, not {from}..{to}",
                self.start,
                self.end()?
            );
        }
        self.layers
            .iter()
            .map(|t| t.narrow(0, from - self.start, to - from))
            .collect()
    }
}

/// The confirmed-prefix hidden states of ONE sequence, or the tail of them a draft reads.
///
/// A model composes one of these and drives it from its layer loop. It is off by default,
/// and then `layers_for` returns nothing, so the loop is byte-identical to the
/// non-speculative path.
///
/// Each captured forward truncates the buffer to its start position before appending. That
/// seeds the prompt on prefill, extends it on each decode, and drops rejected drafts on the
/// next verify forward. Positions alone cannot tell that rollback from a *different* sequence
/// running at a smaller offset, so the buffer is keyed by sequence id: a change of owner
/// reseeds it instead of splicing two sequences' rows together, and a multi-sequence forward —
/// which is never captured — drops it.
///
/// Rows append in place into a buffer sized for the draft's reach, so a decode step costs the
/// rows it adds, and the buffer stays the size of what the draft reads rather than of the
/// context.
#[derive(Default)]
pub struct HiddenPrefixCapture {
    enabled: AtomicBool,
    plan: Mutex<CapturePlan>,
    state: Mutex<CaptureState>,
}

#[derive(Default)]
struct CapturePlan {
    layers: Vec<usize>,
    /// Positions before its anchor a draft may read. `None` keeps the whole prefix.
    retain: Option<usize>,
}

#[derive(Default)]
struct CaptureState {
    /// One `[capacity, hidden]` buffer per captured layer; rows `..len` hold positions
    /// `start..start + len`.
    bufs: Vec<Tensor>,
    start: usize,
    len: usize,
    /// The sequence the buffer describes.
    owner: Option<usize>,
    /// The single sequence the next forward runs; `None` when it runs several.
    forward: Option<usize>,
}

impl CaptureState {
    fn drop_rows(&mut self) {
        self.bufs.clear();
        self.start = 0;
        self.len = 0;
        self.owner = None;
    }
}

/// Smallest buffer an unbounded capture allocates, in rows.
const MIN_ROWS: usize = 256;

impl HiddenPrefixCapture {
    /// Capture `layers` (a draft checkpoint's `target_layer_ids`), keeping the last `retain`
    /// positions, or all of them for `None`. An empty list turns capture off. Either way the
    /// buffer is dropped, so no sequence inherits another's rows.
    pub fn set_layers(&self, layers: Vec<usize>, retain: Option<usize>) {
        self.enabled.store(!layers.is_empty(), Ordering::Relaxed);
        if let Ok(mut plan) = self.plan.lock() {
            *plan = CapturePlan { layers, retain };
        }
        if let Ok(mut state) = self.state.lock() {
            state.drop_rows();
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
                .plan
                .lock()
                .map(|plan| plan.layers.clone())
                .unwrap_or_default();
        }
        if let Ok(mut state) = self.state.lock() {
            state.drop_rows();
        }
        Vec::new()
    }

    /// Fold one forward's captured rows into the buffer. `start_pos` is the KV position of the
    /// forward's first token; `this_forward[k]` is the k-th captured layer, shaped
    /// `[1, query_len, hidden]` or `[query_len, hidden]`.
    pub fn fold(&self, start_pos: usize, this_forward: Vec<Tensor>) -> Result<()> {
        let retain = self.plan.lock().map(|plan| plan.retain).unwrap_or(None);
        let Ok(mut state) = self.state.lock() else {
            return Ok(());
        };
        let HiddenWindow {
            start: mut first,
            layers: mut rows,
        } = HiddenWindow::new(start_pos, this_forward)?;
        let Some(mut added) = rows.first().map(|t| t.dim(0)).transpose()? else {
            return Ok(());
        };

        let fits = |buf: &Tensor, new: &Tensor| -> Result<bool> {
            Ok(buf.dtype() == new.dtype()
                && buf.device().same_device(new.device())
                && buf.dim(1)? == new.dim(1)?)
        };
        let mut same_shape = state.bufs.len() == rows.len();
        for (buf, new) in state.bufs.iter().zip(&rows) {
            same_shape &= fits(buf, new)?;
        }
        if state.owner != state.forward || !same_shape {
            state.drop_rows();
        }
        state.owner = state.forward;

        // Rows at or past this forward's start are rejected drafts. A start the buffer cannot
        // reach, behind it or past its end, reseeds: the window then says which positions it
        // holds, and the proposer decides whether that is enough to draft from.
        if !state.bufs.is_empty() && state.start <= first && first <= state.start + state.len {
            state.len = first - state.start;
        } else {
            state.start = first;
            state.len = 0;
        }

        // A forward longer than the reach contributes only its tail.
        if let Some(keep) = retain {
            if added > keep {
                let skip = added - keep;
                for t in rows.iter_mut() {
                    *t = t.narrow(0, skip, keep)?;
                }
                first += skip;
                added = keep;
                state.start = first;
                state.len = 0;
            }
        }

        let capacity = state.bufs.first().map(|b| b.dim(0)).transpose()?;
        let needed = state.len + added;
        match (retain, capacity) {
            (_, Some(cap)) if needed <= cap => {}
            // Bounded and full: slide the newest rows to the front, keeping `keep` in all.
            (Some(keep), Some(_)) => {
                let kept = keep - added;
                let from = state.len - kept.min(state.len);
                let kept = state.len - from;
                for buf in state.bufs.iter() {
                    buf.slice_set(&buf.narrow(0, from, kept)?.copy()?, 0, 0)?;
                }
                state.start += from;
                state.len = kept;
            }
            // First rows, or unbounded and full: allocate, carrying the live rows over.
            (_, cap) => {
                let rows_cap = match retain {
                    Some(keep) => 2 * keep,
                    None => needed.max(2 * cap.unwrap_or(0)).max(MIN_ROWS),
                };
                let live = state.len;
                let mut bufs = Vec::with_capacity(rows.len());
                for (k, new) in rows.iter().enumerate() {
                    let buf = Tensor::zeros((rows_cap, new.dim(1)?), new.dtype(), new.device())?;
                    if let (Some(old), true) = (state.bufs.get(k), live > 0) {
                        buf.slice_set(&old.narrow(0, 0, live)?.contiguous()?, 0, 0)?;
                    }
                    bufs.push(buf);
                }
                state.bufs = bufs;
            }
        }

        let at = state.len;
        for (buf, new) in state.bufs.iter().zip(&rows) {
            buf.slice_set(&new.contiguous()?, 0, at)?;
        }
        state.len += added;
        Ok(())
    }

    /// What the buffer holds, as views into it: read them before the next forward appends.
    /// `None` while it holds nothing.
    pub fn hiddens(&self) -> Option<HiddenWindow> {
        let state = self.state.lock().ok()?;
        if state.len == 0 {
            return None;
        }
        let layers = state
            .bufs
            .iter()
            .map(|buf| buf.narrow(0, 0, state.len))
            .collect::<Result<Vec<_>>>()
            .ok()?;
        Some(HiddenWindow {
            start: state.start,
            layers,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    /// Rows of width 2: column 0 is `tag`, column 1 the row's absolute position, so both a
    /// row's origin and its place stay readable after truncation, sliding and regrowth.
    fn rows(tag: f32, from: usize, n: usize) -> Result<Tensor> {
        let data: Vec<f32> = (from..from + n).flat_map(|p| [tag, p as f32]).collect();
        Tensor::from_vec(data, (n, 2), &Device::Cpu)
    }

    /// `(start, tags, positions)` of the first captured layer.
    fn held(capture: &HiddenPrefixCapture) -> Result<(usize, Vec<f32>, Vec<usize>)> {
        let Some(window) = capture.hiddens() else {
            return Ok((0, Vec::new(), Vec::new()));
        };
        let rows = window.layers[0].to_vec2::<f32>()?;
        Ok((
            window.start,
            rows.iter().map(|r| r[0]).collect(),
            rows.iter().map(|r| r[1] as usize).collect(),
        ))
    }

    fn capturing(retain: Option<usize>, seq: usize) -> HiddenPrefixCapture {
        let capture = HiddenPrefixCapture::default();
        capture.set_layers(vec![3], retain);
        capture.note_forward(&[seq]);
        capture
    }

    #[test]
    fn off_by_default_and_captures_nothing() {
        let capture = HiddenPrefixCapture::default();
        assert!(capture.layers_for(1).is_empty());
        assert!(capture.hiddens().is_none());
    }

    #[test]
    fn one_sequence_seeds_extends_and_rolls_back() -> Result<()> {
        let capture = capturing(None, 7);
        assert_eq!(capture.layers_for(1), vec![3]);

        capture.fold(0, vec![rows(1.0, 0, 4)?])?; // prefill
        capture.fold(4, vec![rows(2.0, 4, 3)?])?; // verify forward: anchor + two drafts
        assert_eq!(held(&capture)?.1, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);

        // One draft was rejected: the next forward starts at 6 and drops the stale row.
        capture.fold(6, vec![rows(3.0, 6, 1)?])?;
        let (start, tags, positions) = held(&capture)?;
        assert_eq!(start, 0);
        assert_eq!(tags, vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 3.0]);
        assert_eq!(positions, (0..7).collect::<Vec<_>>());
        Ok(())
    }

    /// A is captured, then B runs at a smaller offset. By position alone that is a rollback, and
    /// B would draft from A's hidden states for the rest of its life. The owner check reseeds
    /// instead, and the window says it starts at B's position, not at 0.
    #[test]
    fn another_sequence_never_inherits_the_buffer() -> Result<()> {
        let capture = capturing(None, 1);
        capture.fold(0, vec![rows(1.0, 0, 1000)?])?;

        capture.note_forward(&[2]);
        capture.fold(500, vec![rows(2.0, 500, 1)?])?;
        assert_eq!(
            held(&capture)?,
            (500, vec![2.0], vec![500]),
            "B spliced onto A's rows"
        );

        // B's own prefill from 0 is a legitimate seed.
        capture.fold(0, vec![rows(2.0, 0, 8)?])?;
        let (start, tags, _) = held(&capture)?;
        assert_eq!((start, tags.len()), (0, 8));
        Ok(())
    }

    #[test]
    fn a_multi_sequence_forward_drops_the_buffer() -> Result<()> {
        let capture = capturing(None, 1);
        capture.fold(0, vec![rows(1.0, 0, 16)?])?;

        capture.note_forward(&[1, 2]);
        assert!(capture.layers_for(2).is_empty());
        assert!(
            capture.hiddens().is_none(),
            "a frozen buffer outlived a batched forward"
        );

        // The survivor resumes alone mid-sequence: the window starts where it resumed.
        capture.note_forward(&[1]);
        capture.fold(17, vec![rows(1.0, 17, 1)?])?;
        assert_eq!(held(&capture)?, (17, vec![1.0], vec![17]));
        Ok(())
    }

    #[test]
    fn a_leading_batch_dim_of_one_is_squeezed() -> Result<()> {
        let capture = capturing(None, 1);
        capture.fold(0, vec![rows(1.0, 0, 5)?.unsqueeze(0)?])?;
        assert_eq!(capture.hiddens().unwrap().layers[0].dims(), &[5, 2]);
        Ok(())
    }

    /// A bounded capture holds the newest `retain` positions however long the context grows,
    /// and every row it holds is still the row for its position.
    #[test]
    fn a_bounded_capture_keeps_the_newest_positions() -> Result<()> {
        const KEEP: usize = 4;
        let capture = capturing(Some(KEEP), 1);

        // A prefill longer than the reach contributes only its tail.
        capture.fold(0, vec![rows(1.0, 0, 10)?])?;
        assert_eq!(held(&capture)?, (6, vec![1.0; 4], vec![6, 7, 8, 9]));

        // Decode well past the buffer's capacity, which forces it to slide more than once.
        for pos in 10..40 {
            capture.fold(pos, vec![rows(2.0, pos, 1)?])?;
            let (start, _, positions) = held(&capture)?;
            assert!(positions.len() >= KEEP, "fewer than the reach at {pos}");
            assert_eq!(positions, (start..=pos).collect::<Vec<_>>(), "gap at {pos}");
        }
        Ok(())
    }

    /// Rejected drafts roll back inside a bounded window exactly as in an unbounded one.
    #[test]
    fn a_bounded_capture_rolls_back() -> Result<()> {
        let capture = capturing(Some(6), 1);
        capture.fold(0, vec![rows(1.0, 0, 20)?])?;
        capture.fold(20, vec![rows(2.0, 20, 4)?])?; // verify: anchor + three drafts
        capture.fold(22, vec![rows(3.0, 22, 4)?])?; // two were rejected
        let (start, tags, positions) = held(&capture)?;
        assert_eq!(positions, (start..26).collect::<Vec<_>>());
        assert_eq!(&tags[tags.len() - 6..], &[2.0, 2.0, 3.0, 3.0, 3.0, 3.0]);

        // A forward that starts before the window cannot be bridged: it reseeds.
        capture.fold(3, vec![rows(4.0, 3, 1)?])?;
        assert_eq!(held(&capture)?, (3, vec![4.0], vec![3]));
        Ok(())
    }

    #[test]
    fn an_unbounded_capture_grows_without_losing_rows() -> Result<()> {
        let capture = capturing(None, 1);
        capture.fold(0, vec![rows(1.0, 0, MIN_ROWS - 1)?])?;
        for pos in MIN_ROWS - 1..3 * MIN_ROWS {
            capture.fold(pos, vec![rows(2.0, pos, 1)?])?;
        }
        let (start, _, positions) = held(&capture)?;
        assert_eq!(start, 0);
        assert_eq!(positions, (0..3 * MIN_ROWS).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn a_window_refuses_rows_it_does_not_hold() -> Result<()> {
        let window = HiddenWindow::new(10, vec![rows(1.0, 10, 5)?])?;
        assert_eq!(window.end()?, 15);
        assert_eq!(window.rows(12, 15)?[0].dim(0)?, 3);
        assert!(window.rows(9, 12).is_err());
        assert!(window.rows(12, 16).is_err());
        Ok(())
    }
}
