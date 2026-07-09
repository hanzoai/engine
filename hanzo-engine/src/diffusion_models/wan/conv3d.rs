#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use hanzo_ml::{Result, Tensor};
use hanzo_nn::{Conv2d, Conv2dConfig};
use hanzo_quant::{Convolution, ShardedVarBuilder};

// Wan stashes the trailing CACHE_T frames of every causal-conv input to seed the next
// autoregressive window; 2 frames is the left context a kernel-3 temporal conv needs.
pub const CACHE_T: usize = 2;

// 3D causal convolution decomposed into a temporal loop of conv2d. hanzo-ml has no conv3d on
// any backend, but the Wan temporal kernel is tiny (1 or 3): slice the kernel along time into
// `kt` 2D kernels and sum strided conv2d over the causally left-padded frames. Causal left pad
// = effective_kernel - stride = 2*padding_t for Wan's symmetric configs (engine pitfall #9);
// streaming windows replace those zeros with cached prior frames.
pub struct CausalConv3d {
    taps: Vec<Conv2d>, // one bias-less conv2d per temporal kernel offset
    bias: Option<Tensor>,
    stride_t: usize,
    pad_t: usize,
}

impl CausalConv3d {
    pub fn new(
        in_c: usize,
        out_c: usize,
        kernel: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let (kt, kh, kw) = kernel;
        let w = vb.get((out_c, in_c, kt, kh, kw), "weight")?;
        let bias = if bias {
            Some(vb.get(out_c, "bias")?)
        } else {
            None
        };
        Self::from_weight(w, bias, stride, padding)
    }

    fn from_weight(
        w: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Result<Self> {
        let kt = w.dim(2)?;
        // Wan only uses square spatial kernels/pads, so one Conv2dConfig covers h and w.
        debug_assert_eq!(
            padding.1, padding.2,
            "wan causal conv assumes pad_h == pad_w"
        );
        let cfg = Conv2dConfig {
            padding: padding.1,
            stride: stride.1,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let mut taps = Vec::with_capacity(kt);
        for dt in 0..kt {
            let w2d = w.narrow(2, dt, 1)?.squeeze(2)?.contiguous()?;
            taps.push(Conv2d::new(w2d, None, cfg));
        }
        Ok(Self {
            taps,
            bias,
            stride_t: stride.0,
            pad_t: 2 * padding.0,
        })
    }

    // `cache_x`: trailing frames from the previous streaming window, or None for a full pass.
    pub fn forward(&self, x: &Tensor, cache_x: Option<&Tensor>) -> Result<Tensor> {
        let mut left = self.pad_t;
        let x = match cache_x {
            Some(c) if self.pad_t > 0 => {
                left = self.pad_t.saturating_sub(c.dim(2)?);
                Tensor::cat(&[c, x], 2)?
            }
            _ => x.clone(),
        };
        let x = if left > 0 {
            x.pad_with_zeros(2, left, 0)?
        } else {
            x
        };
        self.conv_temporal(&x)
    }

    fn conv_temporal(&self, x: &Tensor) -> Result<Tensor> {
        let kt = self.taps.len();
        let t_in = x.dim(2)?;
        let t_out = (t_in - kt) / self.stride_t + 1;
        let mut frames = Vec::with_capacity(t_out);
        for to in 0..t_out {
            let base = to * self.stride_t;
            let mut acc: Option<Tensor> = None;
            for (dt, tap) in self.taps.iter().enumerate() {
                let frame = x.narrow(2, base + dt, 1)?.squeeze(2)?;
                let y = Convolution.forward_2d(tap, &frame)?;
                acc = Some(match acc {
                    None => y,
                    Some(a) => (a + y)?,
                });
            }
            frames.push(acc.unwrap().unsqueeze(2)?);
        }
        let out = Tensor::cat(&frames, 2)?;
        match &self.bias {
            Some(b) => out.broadcast_add(&b.reshape((1, b.dim(0)?, 1, 1, 1))?),
            None => Ok(out),
        }
    }
}

// Streaming temporal cache for the chunked VAE: one slot per caching conv, consumed in
// fixed module-traversal order via `cursor` (mirrors Wan's feat_cache/feat_idx). `rewind`
// resets the cursor at the start of each autoregressive window.
pub struct FeatCache {
    slots: Vec<Slot>,
    cursor: usize,
}

#[derive(Clone)]
pub enum Slot {
    Empty,
    Rep, // upsample3d "replicate" sentinel: first window emits no temporal doubling
    Frames(Tensor),
}

impl Default for FeatCache {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatCache {
    // Slots grow on demand as convs are visited; counting them up front is brittle.
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            cursor: 0,
        }
    }

    pub fn rewind(&mut self) {
        self.cursor = 0;
    }

    pub fn advance(&mut self) -> usize {
        let i = self.cursor;
        self.cursor += 1;
        while self.slots.len() <= i {
            self.slots.push(Slot::Empty);
        }
        i
    }

    pub fn slot(&self, i: usize) -> &Slot {
        &self.slots[i]
    }

    pub fn set(&mut self, i: usize, s: Slot) {
        self.slots[i] = s;
    }

    // Per-conv step for kernel-3 causal convs: returns the previous window's cached frames to
    // feed the conv, and stashes this input's trailing CACHE_T frames for the next window. A
    // single-frame window merges the previous tail so the conv always sees >=2 left frames.
    pub fn step(&mut self, x: &Tensor) -> Result<Option<Tensor>> {
        let i = self.advance();
        let prev = match &self.slots[i] {
            Slot::Frames(t) => Some(t.clone()),
            _ => None,
        };
        let t = x.dim(2)?;
        let take = CACHE_T.min(t);
        let mut nc = x.narrow(2, t - take, take)?.contiguous()?;
        if take < CACHE_T {
            if let Some(p) = &prev {
                let pt = p.dim(2)?;
                nc = Tensor::cat(&[&p.narrow(2, pt - 1, 1)?, &nc], 2)?;
            }
        }
        self.slots[i] = Slot::Frames(nc);
        Ok(prev)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{DType, Device};

    // kernel (1,3,3): temporal size 1 collapses the decomposition to a plain per-frame conv2d,
    // so every output frame must equal conv2d applied to that input frame independently.
    #[test]
    fn temporal_one_matches_per_frame_conv2d() -> Result<()> {
        let dev = Device::Cpu;
        let (b, cin, cout, t, hw) = (1usize, 2usize, 4usize, 3usize, 5usize);
        let w = Tensor::randn(0f64, 0.3, (cout, cin, 1, 3, 3), &dev)?.to_dtype(DType::F32)?;
        let bias = Tensor::randn(0f64, 0.3, cout, &dev)?.to_dtype(DType::F32)?;
        let conv = CausalConv3d::from_weight(w.clone(), Some(bias.clone()), (1, 1, 1), (0, 1, 1))?;
        let x = Tensor::randn(0f64, 1.0, (b, cin, t, hw, hw), &dev)?.to_dtype(DType::F32)?;
        let out = conv.forward(&x, None)?;
        assert_eq!(out.dims(), &[b, cout, t, hw, hw]);

        let cfg = Conv2dConfig {
            padding: 1,
            ..Default::default()
        };
        let ref_conv = Conv2d::new(w.squeeze(2)?, Some(bias), cfg);
        for f in 0..t {
            let frame = x.narrow(2, f, 1)?.squeeze(2)?;
            let want = Convolution.forward_2d(&ref_conv, &frame)?;
            let got = out.narrow(2, f, 1)?.squeeze(2)?;
            let diff = (got - want)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 1e-4, "frame {f} mismatch, max abs diff {diff}");
        }
        Ok(())
    }

    // kernel (3,1,1) padding (1,0,0): causal left-pad of 2 zeros then a temporal sum kernel.
    // input [1,2,3,4] -> padded [0,0,1,2,3,4] -> sliding window of 3 -> [1,3,6,9].
    #[test]
    fn temporal_causal_left_pad_sum() -> Result<()> {
        let dev = Device::Cpu;
        let w = Tensor::ones((1, 1, 3, 1, 1), DType::F32, &dev)?;
        let conv = CausalConv3d::from_weight(w, None, (1, 1, 1), (1, 0, 0))?;
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], (1, 1, 4, 1, 1), &dev)?;
        let out = conv.forward(&x, None)?;
        assert_eq!(out.dims(), &[1, 1, 4, 1, 1]);
        let got = out.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(got, vec![1.0, 3.0, 6.0, 9.0]);
        Ok(())
    }
}
