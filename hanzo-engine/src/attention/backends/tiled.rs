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
    /// eager path it replaces. Both sit a few thousandths out because `MatMul` runs f16 on CPU.
    const TOLERANCE: f32 = 1e-3;

    fn params(n_kv_groups: usize, head_dim: usize, softcap: Option<f32>) -> SdpaParams {
        SdpaParams {
            n_kv_groups,
            softcap,
            softmax_scale: 1.0 / (head_dim as f32).sqrt(),
            sliding_window: None,
            sinks: None,
        }
    }

    fn max_abs_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        let a: Vec<f32> = a.flatten_all()?.to_vec1()?;
        let b: Vec<f32> = b.flatten_all()?.to_vec1()?;
        assert!(
            a.iter().all(|x| x.is_finite()),
            "attention output is not finite"
        );
        Ok(a.iter()
            .zip(b.iter())
            .fold(0f32, |acc, (x, y)| acc.max((x - y).abs())))
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
        let eager = max_abs_diff(&eager_reference(q, k, v, mask, sdpa_params)?, &oracle)?;
        let tiled = tiled_sdpa(q, k, v, mask, sdpa_params, q_tile, kv_tile)?;
        assert_eq!(tiled.dims(), oracle.dims(), "{case}");
        let diff = max_abs_diff(&tiled, &oracle)?;
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
        for (b, hq, hkv, d, q_len, kv_len) in shapes {
            for softcap in [None, Some(30.0f32)] {
                let sdpa_params = params(hq / hkv, d, softcap);
                let q = Tensor::randn(0f32, 1., (b, hq, q_len, d), &device)?;
                let k = Tensor::randn(0f32, 1., (b, hkv, kv_len, d), &device)?;
                let v = Tensor::randn(0f32, 1., (b, hkv, kv_len, d), &device)?;
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
        let q = Tensor::randn(0f32, 1., (b, hq, q_len, d), &device)?;
        let k = Tensor::randn(0f32, 1., (b, hkv, kv_len, d), &device)?;
        let v = Tensor::randn(0f32, 1., (b, hkv, kv_len, d), &device)?;
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
        let q = Tensor::randn(0f32, 1., (b, hq, q_len, d), &device)?;
        let k = Tensor::randn(0f32, 1., (b, hkv, kv_len, d), &device)?;
        let v = Tensor::randn(0f32, 1., (b, hkv, kv_len, d), &device)?;
        let mask =
            prefix_causal_mask(q_len, kv_len, None, &device)?.reshape((1, 1, q_len, kv_len))?;
        assert_matches_eager(
            &q,
            &k,
            &v,
            Some(&mask),
            &sdpa_params,
            16,
            64,
            "broadcast mask",
        )
    }
}
