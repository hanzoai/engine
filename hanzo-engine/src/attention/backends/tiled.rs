use hanzo_ml::{DType, Result, Tensor, D};
use hanzo_quant::MatMul;

use crate::attention::{repeat_kv, SdpaParams};

use super::maybe_synchronize;

/// Finite floor for the running row maximum. Attention logits never come near it, so it never moves a
/// real maximum; it is there so a fully masked tile subtracts a finite number instead of -inf - -inf.
const ROW_MAX_FLOOR: f32 = -1e30;

/// Eager attention over [q_tile, kv_tile] score blocks, combined with an online softmax. Same
/// arithmetic as `naive_sdpa` with the softmax reassociated across key tiles, so the block stays the
/// same size whatever the context length instead of growing with it.
pub(crate) fn tiled_sdpa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sdpa_params: &SdpaParams,
    q_tile: usize,
    kv_tile: usize,
) -> Result<Tensor> {
    maybe_synchronize(q.device())?;

    let (b_sz, n_heads, q_len, _) = q.dims4()?;
    let kv_len = k.dim(2)?;
    let v_head_dim = v.dim(3)?;
    let dtype = q.dtype();
    let device = q.device();

    let mut out = Vec::with_capacity(q_len.div_ceil(q_tile));
    for q_start in (0..q_len).step_by(q_tile) {
        let q_rows = q_tile.min(q_len - q_start);
        let q_blk = q.narrow(2, q_start, q_rows)?.contiguous()?;

        let mut acc = Tensor::zeros((b_sz, n_heads, q_rows, v_head_dim), DType::F32, device)?;
        let mut row_sum = Tensor::zeros((b_sz, n_heads, q_rows, 1), DType::F32, device)?;
        let mut row_max = Tensor::full(ROW_MAX_FLOOR, (b_sz, n_heads, q_rows, 1), device)?;

        for kv_start in (0..kv_len).step_by(kv_tile) {
            let kv_cols = kv_tile.min(kv_len - kv_start);
            let k_blk = repeat_kv(k.narrow(2, kv_start, kv_cols)?, sdpa_params.n_kv_groups)?
                .contiguous()?;
            let v_blk = repeat_kv(v.narrow(2, kv_start, kv_cols)?, sdpa_params.n_kv_groups)?
                .contiguous()?;

            let mut scores =
                MatMul.matmul_affine_mul(&q_blk, &k_blk.t()?, sdpa_params.softmax_scale.into())?;
            if let Some(softcap) = sdpa_params.softcap {
                scores = (scores / softcap as f64)?;
                scores = scores.tanh()?;
                scores = (scores * softcap as f64)?;
            }
            if let Some(mask) = mask {
                let tile = mask_tile(mask, q_start, q_rows, kv_start, kv_cols)?;
                scores = scores.broadcast_add(&tile)?;
            }
            let scores = scores.to_dtype(DType::F32)?;

            let next_max = row_max.maximum(&scores.max_keepdim(D::Minus1)?)?;
            let rescale = row_max.sub(&next_max)?.exp()?;
            let weights = scores.broadcast_sub(&next_max)?.exp()?;
            let context = MatMul
                .matmul(&weights.to_dtype(dtype)?, &v_blk)?
                .to_dtype(DType::F32)?;

            acc = acc.broadcast_mul(&rescale)?.add(&context)?;
            row_sum = row_sum
                .mul(&rescale)?
                .add(&weights.sum_keepdim(D::Minus1)?)?;
            row_max = next_max;
        }

        out.push(acc.broadcast_div(&row_sum)?.to_dtype(dtype)?);
    }

    Tensor::cat(&out, 2)
}

/// A mask dimension of 1 broadcasts over the whole axis, so only a materialized axis is narrowed.
fn mask_tile(
    mask: &Tensor,
    q_start: usize,
    q_rows: usize,
    kv_start: usize,
    kv_cols: usize,
) -> Result<Tensor> {
    let rank = mask.rank();
    let (q_dim, kv_dim) = (rank - 2, rank - 1);
    let tile = if mask.dim(q_dim)? == 1 {
        mask.clone()
    } else {
        mask.narrow(q_dim, q_start, q_rows)?
    };
    if tile.dim(kv_dim)? == 1 {
        Ok(tile)
    } else {
        tile.narrow(kv_dim, kv_start, kv_cols)
    }
}

#[cfg(test)]
mod tests {
    use super::tiled_sdpa;
    use crate::attention::{naive_sdpa, repeat_kv, SdpaParams};
    use hanzo_ml::{Device, Result, Tensor};

    /// How much further from the f32 oracle the key split is allowed to land than the whole-block
    /// eager path it replaces, as RMS over the output. Both sit a few ten-thousandths out because
    /// `MatMul` runs f16 on CPU. Over 18 shape/mask/softcap variants x 128 draws the split is never
    /// the worse of the two -- it runs 5-46% under eager, and the least favourable single draw still
    /// sat 2.2e-5 under. This allowance is the round number above that, at least 9.6 standard
    /// deviations clear of every variant's spread.
    const TOLERANCE: f32 = 1e-4;

    /// Deterministic standard-normal noise: splitmix64 -> uniform pairs -> Box-Muller.
    ///
    /// This stack's CPU `Device::set_seed` is a no-op, so the random-tensor constructors draw fresh
    /// values on every run, and these tests assert a numeric bound. Naming the data makes a failure
    /// here reproduce forever.
    fn normal(shape: (usize, usize, usize, usize), seed: u64, device: &Device) -> Result<Tensor> {
        let (a, b, c, d) = shape;
        let n = a * b * c * d;
        let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // (0, 1]: Box-Muller's log must never see zero.
            ((z >> 11) as f64 + 1.0) / ((1u64 << 53) as f64 + 1.0)
        };
        let mut values = Vec::with_capacity(n);
        while values.len() < n {
            let (u1, u2) = (next(), next());
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = std::f64::consts::TAU * u2;
            values.push((r * theta.cos()) as f32);
            if values.len() < n {
                values.push((r * theta.sin()) as f32);
            }
        }
        Tensor::from_vec(values, shape, device)
    }

    fn params(n_kv_groups: usize, head_dim: usize, softcap: Option<f32>) -> SdpaParams {
        SdpaParams {
            n_kv_groups,
            softcap,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        }
    }

    /// RMS of the elementwise difference. A maximum reads one element in ~10^5, so its draw-to-draw
    /// spread (~1e-3) buries the accuracy difference being asserted; the mean square reads every
    /// element and spreads ~2e-6.
    fn rms_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = b.flatten_all()?.to_vec1()?;
        assert!(
            a.iter().all(|x| x.is_finite()),
            "attention output is not finite"
        );
        let square_sum: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| f64::from(x - y).powi(2))
            .sum();
        Ok((square_sum / a.len() as f64).sqrt() as f32)
    }

    fn eager_reference(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;
        naive_sdpa(q, &k, &v, mask, sdpa_params)
    }

    /// The same attention in plain f32: `Tensor::matmul` keeps the dtype it is given, where `MatMul`
    /// drops to f16 on CPU. Both real paths are measured against this, since the split changes the
    /// reduction the f16 value matmul performs and is the more accurate of the two.
    fn f32_oracle(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
    ) -> Result<Tensor> {
        let k = repeat_kv(k.clone(), sdpa_params.n_kv_groups)?;
        let v = repeat_kv(v.clone(), sdpa_params.n_kv_groups)?;
        let mut att = (q.matmul(&k.t()?.contiguous()?)? * sdpa_params.softmax_scale as f64)?;
        if let Some(softcap) = sdpa_params.softcap {
            att = ((att / softcap as f64)?.tanh()? * softcap as f64)?;
        }
        if let Some(mask) = mask {
            att = att.broadcast_add(mask)?;
        }
        hanzo_nn::ops::softmax_last_dim(&att)?.matmul(&v)
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_matches_eager(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        sdpa_params: &SdpaParams,
        q_tile: usize,
        kv_tile: usize,
        case: &str,
    ) -> Result<()> {
        let oracle = f32_oracle(q, k, v, mask, sdpa_params)?;
        let eager = rms_diff(&eager_reference(q, k, v, mask, sdpa_params)?, &oracle)?;
        let tiled = tiled_sdpa(q, k, v, mask, sdpa_params, q_tile, kv_tile)?;
        assert_eq!(tiled.dims(), oracle.dims(), "{case}");
        let diff = rms_diff(&tiled, &oracle)?;
        assert!(
            diff <= eager + TOLERANCE,
            "{case}: tiled is {diff} from the f32 oracle, eager is {eager}"
        );
        Ok(())
    }

    /// `0` where key `j` is visible to query `i`, -inf elsewhere, over a prefix of `kv_len - q_len`
    /// keys every query can see.
    fn prefix_causal_mask(
        q_len: usize,
        kv_len: usize,
        window: Option<usize>,
        device: &Device,
    ) -> Result<Tensor> {
        let prefix = kv_len - q_len;
        let mut values = Vec::with_capacity(q_len * kv_len);
        for i in 0..q_len {
            for j in 0..kv_len {
                let pos = prefix + i;
                let future = j > pos;
                let too_old = window.is_some_and(|w| pos >= w && j <= pos - w);
                values.push(if future || too_old {
                    f32::NEG_INFINITY
                } else {
                    0.0
                });
            }
        }
        Tensor::from_vec(values, (q_len, kv_len), device)
    }

    #[test]
    fn tiled_matches_eager_across_shapes() -> Result<()> {
        let device = Device::Cpu;
        // (batch, q heads, kv heads, head dim, q len, kv len); the 24/4 rows carry the GQA group of
        // 6 that keeps Qwen3.5-27B off every paged flash kernel.
        let shapes = [
            (1usize, 24usize, 4usize, 64usize, 96usize, 96usize),
            (1, 24, 4, 64, 37, 333),
            (2, 8, 8, 32, 64, 200),
            (1, 6, 2, 64, 1, 257),
        ];
        for (case, (b, hq, hkv, d, q_len, kv_len)) in shapes.into_iter().enumerate() {
            for softcap in [None, Some(30.0f32)] {
                let sdpa_params = params(hq / hkv, d, softcap);
                let seed = case as u64;
                let q = normal((b, hq, q_len, d), 100 + seed, &device)?;
                let k = normal((b, hkv, kv_len, d), 200 + seed, &device)?;
                let v = normal((b, hkv, kv_len, d), 300 + seed, &device)?;
                let causal = prefix_causal_mask(q_len, kv_len, None, &device)?;
                for mask in [None, Some(&causal)] {
                    let case = format!(
                        "b={b} hq={hq} hkv={hkv} d={d} q={q_len} kv={kv_len} \
                         softcap={softcap:?} mask={}",
                        mask.is_some()
                    );
                    assert_matches_eager(&q, &k, &v, mask, &sdpa_params, 16, 24, &case)?;
                }
            }
        }
        Ok(())
    }

    /// A window narrow enough that the leading key tiles are fully masked for the later queries: the
    /// running maximum has to survive a tile in which every score is -inf.
    #[test]
    fn tiled_matches_eager_with_fully_masked_tiles() -> Result<()> {
        let device = Device::Cpu;
        let (b, hq, hkv, d, q_len, kv_len) = (1usize, 8usize, 2usize, 32usize, 128usize, 512usize);
        let sdpa_params = params(hq / hkv, d, None);
        let q = normal((b, hq, q_len, d), 400, &device)?;
        let k = normal((b, hkv, kv_len, d), 500, &device)?;
        let v = normal((b, hkv, kv_len, d), 600, &device)?;
        let mask = prefix_causal_mask(q_len, kv_len, Some(64), &device)?;
        assert_matches_eager(
            &q,
            &k,
            &v,
            Some(&mask),
            &sdpa_params,
            32,
            32,
            "sliding window",
        )
    }

    /// A rank-4 mask whose batch and head axes broadcast is the shape the paged prefill hands down.
    #[test]
    fn tiled_narrows_a_broadcast_mask() -> Result<()> {
        let device = Device::Cpu;
        let (b, hq, hkv, d, q_len, kv_len) = (2usize, 8usize, 4usize, 32usize, 48usize, 160usize);
        let sdpa_params = params(hq / hkv, d, None);
        let mask =
            prefix_causal_mask(q_len, kv_len, None, &device)?.reshape((1, 1, q_len, kv_len))?;
        // Eight named draws, not one: the key split's error depends on the data, so a single
        // sample says little about whether tiling costs precision.
        for seed in 0..8u64 {
            let q = normal((b, hq, q_len, d), 700 + seed, &device)?;
            let k = normal((b, hkv, kv_len, d), 800 + seed, &device)?;
            let v = normal((b, hkv, kv_len, d), 900 + seed, &device)?;
            assert_matches_eager(
                &q,
                &k,
                &v,
                Some(&mask),
                &sdpa_params,
                16,
                64,
                &format!("broadcast mask seed={seed}"),
            )?;
        }
        Ok(())
    }
}
