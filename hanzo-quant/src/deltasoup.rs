//! DeltaSoup: Byzantine-robust aggregation of contributor deltas.
//!
//! Five methods, all coordinate-wise on a stack of same-shape deltas:
//!
//! - [`Method::Mean`]                       — plain mean.
//! - [`Method::Median`]                     — coordinate-wise median.
//! - [`Method::TrimmedMean { trim }`]       — drop top+bottom `floor(trim * N)`
//!   per coordinate then mean. Per the task brief, for `N >= 4` with the
//!   conventional `trim=1` the implementation drops exactly one max and one
//!   min and means the rest; for `N < 4` it falls back to plain mean.
//! - [`Method::Krum { f }`]                 — Blanchard et al. 2017. Picks the
//!   single delta whose sum of `n-f-2` smallest squared L2 distances to peers
//!   is minimal.
//! - [`Method::MultiKrum { f, m }`]         — same scoring; means the `m`
//!   best-scoring deltas.
//!
//! **v1 scope:** just the math. Reputation, DP noise, and reward distribution
//! from `deltasoup.py` are not ported.

use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Method {
    Mean,
    Median,
    /// Trimmed mean. `trim` is the fraction trimmed from *each* tail; e.g.
    /// `trim=0.1, N=10` -> drop 1 from each end. As a special case (the one
    /// the task brief calls out): with `trim` resolving to >= 1 element per
    /// tail and `N >= 4`, you drop one max and one min and mean the rest.
    /// With `N < 4` falls back to mean.
    TrimmedMean {
        trim: f32,
    },
    /// Krum: select the single delta with the minimum Krum score.
    /// `f` = max assumed Byzantine workers. Requires `N >= 2*f + 3`.
    Krum {
        f: usize,
    },
    /// Multi-Krum: mean of the top `m` deltas by Krum score.
    MultiKrum {
        f: usize,
        m: usize,
    },
}

/// Aggregate `deltas` according to `method`. All deltas must have the same
/// shape and dtype-compatible-with-f32 (we cast internally). Returns an f32
/// tensor on the same device as the first input.
pub fn aggregate(method: Method, deltas: &[Tensor]) -> Result<Tensor> {
    if deltas.is_empty() {
        return Err(Error::Empty("deltasoup::aggregate: no deltas"));
    }
    let dev = deltas[0].device().clone();
    let shape = deltas[0].dims().to_vec();
    for d in deltas {
        if d.dims() != shape.as_slice() {
            return Err(Error::ShapeMismatch {
                base: shape.clone(),
                weight: d.dims().to_vec(),
            });
        }
    }

    // Materialize as Vec<Vec<f32>> for coordinate-wise ops. Trades memory for
    // simplicity — these are aggregator-side ops, not hot-path inference.
    let n = deltas.len();
    let numel: usize = shape.iter().product();
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for d in deltas {
        let v: Vec<f32> = d.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        debug_assert_eq!(v.len(), numel);
        rows.push(v);
    }

    let out: Vec<f32> = match method {
        Method::Mean => coord_mean(&rows, numel),
        Method::Median => coord_median(&rows, numel),
        Method::TrimmedMean { trim } => coord_trimmed_mean(&rows, numel, trim),
        Method::Krum { f } => krum(&rows, numel, f, 1)?,
        Method::MultiKrum { f, m } => krum(&rows, numel, f, m)?,
    };

    Ok(Tensor::from_vec(out, shape.as_slice(), &dev)?)
}

fn coord_mean(rows: &[Vec<f32>], numel: usize) -> Vec<f32> {
    let n = rows.len() as f32;
    let mut out = vec![0.0_f32; numel];
    for r in rows {
        for (i, &v) in r.iter().enumerate() {
            out[i] += v;
        }
    }
    for x in &mut out {
        *x /= n;
    }
    out
}

fn coord_median(rows: &[Vec<f32>], numel: usize) -> Vec<f32> {
    let n = rows.len();
    let mut out = Vec::with_capacity(numel);
    let mut col = Vec::with_capacity(n);
    for i in 0..numel {
        col.clear();
        for r in rows {
            col.push(r[i]);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = if n % 2 == 1 {
            col[n / 2]
        } else {
            0.5 * (col[n / 2 - 1] + col[n / 2])
        };
        out.push(m);
    }
    out
}

fn coord_trimmed_mean(rows: &[Vec<f32>], numel: usize, trim: f32) -> Vec<f32> {
    let n = rows.len();
    // Task-brief invariant: N < 4 -> mean.
    if n < 4 {
        return coord_mean(rows, numel);
    }
    // Per-tail trim count: floor(trim*N), but always at least the brief's
    // "drop 1 from each end" if the caller passed a sentinel trim>0 with N>=4.
    let raw = (trim * n as f32).floor() as usize;
    let trim_n = raw.max(1).min((n - 1) / 2);

    let mut out = Vec::with_capacity(numel);
    let mut col = Vec::with_capacity(n);
    let keep = n - 2 * trim_n;
    for i in 0..numel {
        col.clear();
        for r in rows {
            col.push(r[i]);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let slice = &col[trim_n..n - trim_n];
        debug_assert_eq!(slice.len(), keep);
        let s: f32 = slice.iter().sum();
        out.push(s / keep as f32);
    }
    out
}

fn krum(rows: &[Vec<f32>], numel: usize, f: usize, m: usize) -> Result<Vec<f32>> {
    let n = rows.len();
    if n < 2 * f + 3 {
        return Err(Error::NotEnoughDeltas { needed: 2 * f + 3, got: n });
    }
    if m == 0 || m > n {
        return Err(Error::NotEnoughDeltas { needed: m.max(1), got: n });
    }

    // Pairwise squared L2 distances.
    let mut dist = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0_f32;
            for k in 0..numel {
                let d = rows[i][k] - rows[j][k];
                s += d * d;
            }
            dist[i][j] = s;
            dist[j][i] = s;
        }
    }

    // Krum score: sum of n-f-2 smallest non-self distances.
    let take = n.saturating_sub(f + 2);
    let mut scores: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let mut row: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| dist[i][j]).collect();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let s: f32 = row.iter().take(take).sum();
            (i, s)
        })
        .collect();
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    // Mean of top-m by score (lowest scores = best).
    let chosen: Vec<usize> = scores.into_iter().take(m).map(|(i, _)| i).collect();
    let mut out = vec![0.0_f32; numel];
    for &idx in &chosen {
        for k in 0..numel {
            out[k] += rows[idx][k];
        }
    }
    let denom = chosen.len() as f32;
    for x in &mut out {
        *x /= denom;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn t(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), v.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn mean_matches_arith() {
        let deltas = vec![t(&[1.0, 2.0]), t(&[3.0, 4.0]), t(&[5.0, 6.0])];
        let out: Vec<f32> = aggregate(Method::Mean, &deltas).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn median_picks_middle() {
        let deltas = vec![t(&[1.0]), t(&[100.0]), t(&[2.0])];
        let out: Vec<f32> = aggregate(Method::Median, &deltas).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![2.0]); // 100.0 is the Byzantine outlier
    }

    #[test]
    fn trimmed_mean_drops_outliers() {
        // 5 workers, 1 element. trim=0.2 -> drop 1 from each end -> mean of middle 3.
        let deltas = vec![t(&[1.0]), t(&[2.0]), t(&[3.0]), t(&[100.0]), t(&[-100.0])];
        let out: Vec<f32> =
            aggregate(Method::TrimmedMean { trim: 0.2 }, &deltas).unwrap().to_vec1().unwrap();
        // After sorting per coord: [-100, 1, 2, 3, 100], drop ends -> mean(1,2,3) = 2.0.
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn trimmed_mean_falls_back_to_mean_for_small_n() {
        let deltas = vec![t(&[1.0]), t(&[2.0]), t(&[100.0])]; // N=3 < 4
        let out: Vec<f32> =
            aggregate(Method::TrimmedMean { trim: 0.2 }, &deltas).unwrap().to_vec1().unwrap();
        // Mean fallback: 103/3 ~ 34.33...
        assert!((out[0] - 103.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn krum_selects_clustered_delta() {
        // 5 deltas: 4 clustered near origin, 1 wild outlier. n=5, f=1 -> need n>=5. ok.
        let deltas = vec![
            t(&[0.0, 0.0]),
            t(&[0.1, 0.0]),
            t(&[0.0, 0.1]),
            t(&[0.1, 0.1]),
            t(&[1000.0, 1000.0]),
        ];
        let out: Vec<f32> =
            aggregate(Method::Krum { f: 1 }, &deltas).unwrap().to_vec1().unwrap();
        // Should be one of the clustered points (norm small), not the outlier.
        assert!(out[0].abs() < 1.0 && out[1].abs() < 1.0);
    }

    #[test]
    fn multi_krum_averages_top_m() {
        let deltas = vec![
            t(&[0.0]),
            t(&[1.0]),
            t(&[2.0]),
            t(&[3.0]),
            t(&[1000.0]), // Byzantine
        ];
        // n=5, f=1, m=3 -> need n>=5. Average the 3 best.
        let out: Vec<f32> =
            aggregate(Method::MultiKrum { f: 1, m: 3 }, &deltas).unwrap().to_vec1().unwrap();
        // Byzantine should not be selected; best three are some subset of {0,1,2,3}.
        assert!(out[0] < 4.0, "out = {}", out[0]);
    }

    #[test]
    fn shape_mismatch_returns_error() {
        let deltas = vec![t(&[1.0, 2.0]), t(&[1.0])];
        assert!(matches!(
            aggregate(Method::Mean, &deltas),
            Err(Error::ShapeMismatch { .. })
        ));
    }
}
