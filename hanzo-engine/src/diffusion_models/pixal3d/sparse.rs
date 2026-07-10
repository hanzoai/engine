//! Sparse-tensor infrastructure for the TRELLIS SLAT flow + mesh decoder.
//!
//! A `Sparse` value is an active-voxel set: `coords` [N,3] (z,y,x, single batch) plus `feats` [N,C].
//! The ops here reproduce TRELLIS's spconv-backed sparse modules on CPU (B=1): submanifold Conv3d
//! (`SubMConv3d`, output at the same active coords), average-pool `downsample`, its paired nearest
//! `upsample`, `subdivide` (each voxel -> 2^3), the `AbsolutePositionEmbedder`, and the windowed /
//! full self-attention partitioning. N is small (<= ~32k), so gathers are plain index maps.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
use std::collections::HashMap;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::Module;
use hanzo_quant::ShardedVarBuilder;

use crate::layers::linear;

/// Active-voxel sparse tensor (single batch). `coords[i]` is (z,y,x); `feats` is [N, C].
#[derive(Clone)]
pub struct Sparse {
    pub coords: Vec<[i32; 3]>,
    pub feats: Tensor,
}

impl Sparse {
    pub fn new(coords: Vec<[i32; 3]>, feats: Tensor) -> Self {
        Self { coords, feats }
    }

    pub fn n(&self) -> usize {
        self.coords.len()
    }

    /// Replace feats, keep coords.
    pub fn replace(&self, feats: Tensor) -> Self {
        Self {
            coords: self.coords.clone(),
            feats,
        }
    }

    #[allow(dead_code)]
    fn index(&self) -> HashMap<[i32; 3], u32> {
        self.coords
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i as u32))
            .collect()
    }
}

/// Standard `nn.Linear` applied over the [N,C] feats (SparseLinear).
pub struct SparseLinear {
    lin: hanzo_nn::Linear,
}

impl SparseLinear {
    pub fn new(cin: usize, cout: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            lin: linear(cin, cout, vb)?,
        })
    }

    pub fn forward(&self, x: &Sparse) -> Result<Sparse> {
        Ok(x.replace(self.lin.forward(&x.feats)?))
    }

    pub fn forward_feats(&self, feats: &Tensor) -> Result<Tensor> {
        self.lin.forward(feats)
    }
}

/// Submanifold 3x3x3 Conv3d: output lives at the same active coords, gathering the (up to 27)
/// active neighbours `coord + (k-1)`. Weight is stored spconv-layout `[cout, kz, ky, kx, cin]`.
pub struct SubMConv3d {
    /// 27 per-offset weight matrices `[cin, cout]` (offset order kz*9+ky*3+kx).
    wt: Vec<Tensor>,
    bias: Option<Tensor>,
    cin: usize,
    cout: usize,
    ksize: usize,
}

impl SubMConv3d {
    pub fn new(
        cin: usize,
        cout: usize,
        ksize: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let w = vb.get((cout, ksize, ksize, ksize, cin), "weight")?;
        let mut wt = Vec::with_capacity(ksize.pow(3));
        for kz in 0..ksize {
            for ky in 0..ksize {
                for kx in 0..ksize {
                    // [cout, cin] -> [cin, cout] for feats[N,cin] @ w -> [N,cout]
                    let wk = w
                        .narrow(1, kz, 1)?
                        .narrow(2, ky, 1)?
                        .narrow(3, kx, 1)?
                        .reshape((cout, cin))?
                        .t()?
                        .contiguous()?;
                    wt.push(wk);
                }
            }
        }
        let bias = if bias {
            Some(vb.get(cout, "bias")?)
        } else {
            None
        };
        Ok(Self {
            wt,
            bias,
            cin,
            cout,
            ksize,
        })
    }

    /// Build from an explicit spconv-layout weight `[cout,kz,ky,kx,cin]` (for tests / non-vb paths).
    pub fn from_weight(weight: &Tensor, bias: Option<Tensor>) -> Result<Self> {
        let d = weight.dims();
        let (cout, ksize, cin) = (d[0], d[1], d[4]);
        let mut wt = Vec::with_capacity(ksize.pow(3));
        for kz in 0..ksize {
            for ky in 0..ksize {
                for kx in 0..ksize {
                    let wk = weight
                        .narrow(1, kz, 1)?
                        .narrow(2, ky, 1)?
                        .narrow(3, kx, 1)?
                        .reshape((cout, cin))?
                        .t()?
                        .contiguous()?;
                    wt.push(wk);
                }
            }
        }
        Ok(Self {
            wt,
            bias,
            cin,
            cout,
            ksize,
        })
    }

    pub fn forward(&self, x: &Sparse) -> Result<Sparse> {
        Ok(x.replace(self.forward_feats(&x.coords, &x.feats)?))
    }

    pub fn forward_feats(&self, coords: &[[i32; 3]], feats: &Tensor) -> Result<Tensor> {
        let n = coords.len();
        let dev = feats.device();
        let map: HashMap<[i32; 3], u32> = coords
            .iter()
            .enumerate()
            .map(|(i, c)| (*c, i as u32))
            .collect();
        // pad a zero row at index n for missing neighbours.
        let feats_pad = Tensor::cat(&[feats, &Tensor::zeros((1, self.cin), DType::F32, dev)?], 0)?;
        let c = (self.ksize / 2) as i32;
        let mut acc: Option<Tensor> = None;
        for kz in 0..self.ksize {
            for ky in 0..self.ksize {
                for kx in 0..self.ksize {
                    let (dz, dy, dx) = (kz as i32 - c, ky as i32 - c, kx as i32 - c);
                    let mut idx = Vec::with_capacity(n);
                    let mut any = false;
                    for co in coords {
                        match map.get(&[co[0] + dz, co[1] + dy, co[2] + dx]) {
                            Some(&j) => {
                                idx.push(j);
                                any = true;
                            }
                            None => idx.push(n as u32),
                        }
                    }
                    if !any {
                        continue;
                    }
                    let idx_t = Tensor::from_vec(idx, n, dev)?;
                    let g = feats_pad.index_select(&idx_t, 0)?; // [n, cin]
                    let contrib = g.matmul(&self.wt[(kz * self.ksize + ky) * self.ksize + kx])?;
                    acc = Some(match acc {
                        Some(a) => (a + contrib)?,
                        None => contrib,
                    });
                }
            }
        }
        let out = acc.unwrap_or(Tensor::zeros((n, self.cout), DType::F32, dev)?);
        match &self.bias {
            Some(b) => out.broadcast_add(&b.reshape((1, self.cout))?),
            None => Ok(out),
        }
    }
}

/// Average-pool downsample by 2 (single batch). Returns the pooled sparse tensor plus the paired
/// upsample metadata: the original coords and the per-original-voxel group index.
pub struct DownCache {
    pub src_coords: Vec<[i32; 3]>,
    pub idx: Vec<u32>, // original voxel -> pooled group
}

pub fn downsample2(x: &Sparse) -> Result<(Sparse, DownCache)> {
    let n = x.n();
    // code = raveled (z//2, y//2, x//2); dedup ascending == lexicographic (z,y,x).
    let dcoord: Vec<[i32; 3]> = x
        .coords
        .iter()
        .map(|c| [c[0] / 2, c[1] / 2, c[2] / 2])
        .collect();
    let mut uniq: Vec<[i32; 3]> = dcoord.clone();
    uniq.sort_unstable();
    uniq.dedup();
    let pos: HashMap<[i32; 3], u32> = uniq
        .iter()
        .enumerate()
        .map(|(i, c)| (*c, i as u32))
        .collect();
    let idx: Vec<u32> = dcoord.iter().map(|c| pos[c]).collect();
    let m = uniq.len();
    // mean-pool feats by group.
    let dev = x.feats.device();
    let idx_t = Tensor::from_vec(idx.clone(), n, dev)?;
    let cin = x.feats.dim(1)?;
    let idx_exp = idx_t
        .reshape((n, 1))?
        .broadcast_as((n, cin))?
        .contiguous()?;
    let sum = Tensor::zeros((m, cin), DType::F32, dev)?.scatter_add(&idx_exp, &x.feats, 0)?;
    // TRELLIS uses torch.scatter_reduce(mean, include_self=True): the zero-init counts, so the
    // divisor is (group size + 1), not the group size.
    let mut cnt = vec![1f32; m];
    for &g in &idx {
        cnt[g as usize] += 1.0;
    }
    let cnt_t = Tensor::from_vec(cnt, (m, 1), dev)?;
    let feats = sum.broadcast_div(&cnt_t)?;
    let cache = DownCache {
        src_coords: x.coords.clone(),
        idx,
    };
    Ok((Sparse::new(uniq, feats), cache))
}

/// Nearest upsample paired with a prior [`downsample2`]: each original voxel takes its group feature.
pub fn upsample2(x: &Sparse, cache: &DownCache) -> Result<Sparse> {
    let dev = x.feats.device();
    let idx_t = Tensor::from_vec(cache.idx.clone(), cache.idx.len(), dev)?;
    let feats = x.feats.index_select(&idx_t, 0)?;
    Ok(Sparse::new(cache.src_coords.clone(), feats))
}

/// The 8 subdivision offsets, `torch.nonzero(ones[2,2,2])` order (z,y,x lexicographic).
const SUB_OFFSETS: [[i32; 3]; 8] = [
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1],
];

/// Subdivide each voxel into 2^3 children (`coord*2 + offset`), broadcasting feats.
pub fn subdivide(x: &Sparse) -> Result<Sparse> {
    let n = x.n();
    let mut coords = Vec::with_capacity(n * 8);
    for c in &x.coords {
        for o in &SUB_OFFSETS {
            coords.push([c[0] * 2 + o[0], c[1] * 2 + o[1], c[2] * 2 + o[2]]);
        }
    }
    let dev = x.feats.device();
    // feats.unsqueeze(1).expand(n,8,C).flatten(0,1) == repeat_interleave rows by 8.
    let idx: Vec<u32> = (0..n as u32)
        .flat_map(|i| std::iter::repeat_n(i, 8))
        .collect();
    let idx_t = Tensor::from_vec(idx, n * 8, dev)?;
    let feats = x.feats.index_select(&idx_t, 0)?;
    Ok(Sparse::new(coords, feats))
}

/// TRELLIS `AbsolutePositionEmbedder`: per-axis sinusoidal embedding of integer coords, concatenated
/// over (z,y,x) then zero-padded to `channels`.
pub struct AbsolutePositionEmbedder {
    freqs: Vec<f32>,
    channels: usize,
}

impl AbsolutePositionEmbedder {
    pub fn new(channels: usize) -> Self {
        let freq_dim = channels / 3 / 2;
        let freqs = (0..freq_dim)
            .map(|f| 1.0f32 / 10000f32.powf(f as f32 / freq_dim as f32))
            .collect();
        Self { freqs, channels }
    }

    /// coords [N,3] -> [N, channels].
    pub fn forward(&self, coords: &[[i32; 3]], dev: &Device) -> Result<Tensor> {
        let n = coords.len();
        let fd = self.freqs.len();
        let mut data = vec![0f32; n * self.channels];
        for (i, c) in coords.iter().enumerate() {
            let base = i * self.channels;
            for (axis, &p) in c.iter().enumerate() {
                let off = base + axis * 2 * fd;
                for (k, &fr) in self.freqs.iter().enumerate() {
                    let a = p as f32 * fr;
                    data[off + k] = a.sin();
                    data[off + fd + k] = a.cos();
                }
            }
        }
        Tensor::from_vec(data, (n, self.channels), dev)
    }
}

/// TRELLIS `SparseGroupNorm32`: nn.GroupNorm over the [N,C] feats (B=1). Each group of C/G channels
/// is normalized over all voxels + its channels (population var), then per-channel affine.
pub struct SparseGroupNorm32 {
    weight: Tensor, // [C]
    bias: Tensor,   // [C]
    groups: usize,
    channels: usize,
    eps: f64,
}

impl SparseGroupNorm32 {
    pub fn new(groups: usize, channels: usize, vb: ShardedVarBuilder) -> Result<Self> {
        Ok(Self {
            weight: vb.get(channels, "weight")?,
            bias: vb.get(channels, "bias")?,
            groups,
            channels,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Sparse) -> Result<Sparse> {
        Ok(x.replace(self.forward_feats(&x.feats)?))
    }

    pub fn forward_feats(&self, feats: &Tensor) -> Result<Tensor> {
        let n = feats.dim(0)?;
        let cs = self.channels / self.groups;
        let _dev = feats.device();
        // [N, G, cs] -> per-group mean/var over (N, cs).
        let g = feats.reshape((n, self.groups, cs))?;
        let mean = g.mean_keepdim(0)?.mean_keepdim(2)?; // [1, G, 1]
        let xc = g.broadcast_sub(&mean)?;
        let var = xc.sqr()?.mean_keepdim(0)?.mean_keepdim(2)?; // [1, G, 1]
        let norm = xc
            .broadcast_div(&(var + self.eps)?.sqrt()?)?
            .reshape((n, self.channels))?;
        norm.broadcast_mul(&self.weight.reshape((1, self.channels))?)?
            .broadcast_add(&self.bias.reshape((1, self.channels))?)
    }
}

/// Window partition (TRELLIS `calc_window_partition`, single batch, shift): returns per-window voxel
/// index groups. Window id = raveled (z+sh)//ws, (y+sh)//ws, (x+sh)//ws over the used window grid.
pub fn window_partition(coords: &[[i32; 3]], window: i32, shift: i32) -> Vec<Vec<u32>> {
    let sc: Vec<[i32; 3]> = coords
        .iter()
        .map(|c| {
            [
                (c[0] + shift) / window,
                (c[1] + shift) / window,
                (c[2] + shift) / window,
            ]
        })
        .collect();
    // group by window cell, preserving first-seen order is irrelevant (attention is permutation
    // equivariant); we only need each window's member set.
    let mut groups: HashMap<[i32; 3], Vec<u32>> = HashMap::new();
    for (i, w) in sc.iter().enumerate() {
        groups.entry(*w).or_default().push(i as u32);
    }
    groups.into_values().collect()
}

/// Cosine similarity + max|d| + mse between two flat tensors (parity metric).
#[cfg(test)]
pub fn cos_stats(a: &Tensor, b: &Tensor) -> (f64, f64, f64) {
    let a = a.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let b = b.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let (mut dot, mut na, mut nb, mut se, mut mx) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(&b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
        se += (x - y).powi(2);
        mx = mx.max((x - y).abs());
    }
    (dot / (na.sqrt() * nb.sqrt()), mx, se / a.len() as f64)
}

#[cfg(test)]
fn coords_from(t: &Tensor) -> Vec<[i32; 3]> {
    // fixtures store coords as [N,4] float (b,z,y,x).
    let v = t.to_vec2::<f32>().unwrap();
    v.iter()
        .map(|r| [r[1] as i32, r[2] as i32, r[3] as i32])
        .collect()
}

// TRELLIS_FIX=/path/to/oracle/fixtures cargo test -p hanzo-engine pixal3d::sparse -- --nocapture
#[cfg(test)]
mod tests {
    use super::*;

    fn fix(name: &str) -> HashMap<String, Tensor> {
        let dir = std::env::var("TRELLIS_FIX").expect("set TRELLIS_FIX to oracle/fixtures");
        hanzo_ml::safetensors::load(format!("{dir}/{name}"), &Device::Cpu).unwrap()
    }

    #[test]
    #[ignore = "needs TRELLIS_FIX"]
    fn submconv_parity() {
        let io = fix("submconv_io.safetensors");
        let coords = coords_from(&io["coords"]);
        let conv = SubMConv3d::from_weight(&io["weight"], Some(io["bias"].clone())).unwrap();
        let out = conv.forward_feats(&coords, &io["feats"]).unwrap();
        let (cos, mx, mse) = cos_stats(&out, &io["out"]);
        println!("submconv cos={cos:.8} max|d|={mx:.3e} mse={mse:.3e}");
        assert!(cos > 0.9999, "cos {cos}");
    }

    #[test]
    #[ignore = "needs TRELLIS_FIX"]
    fn downup_parity() {
        let io = fix("downup_io.safetensors");
        let x = Sparse::new(coords_from(&io["coords"]), io["feats"].clone());
        let (down, cache) = downsample2(&x).unwrap();
        assert_eq!(down.n(), io["down_coords"].dim(0).unwrap());
        let (cd, _, _) = cos_stats(&down.feats, &io["down_feats"]);
        let up = upsample2(&down, &cache).unwrap();
        let (cu, mxu, _) = cos_stats(&up.feats, &io["up_feats"]);
        println!("down cos={cd:.8}  up cos={cu:.8} max|d|={mxu:.3e}");
        assert!(cd > 0.9999 && cu > 0.9999, "down {cd} up {cu}");
    }

    #[test]
    #[ignore = "needs TRELLIS_FIX"]
    fn subdivide_parity() {
        let io = fix("subdivide_io.safetensors");
        let x = Sparse::new(coords_from(&io["coords"]), io["feats"].clone());
        let sub = subdivide(&x).unwrap();
        assert_eq!(sub.n(), io["sub_coords"].dim(0).unwrap());
        // coords must match exactly (order-sensitive)
        let ref_c = coords_from(&io["sub_coords"]);
        assert_eq!(sub.coords, ref_c, "subdivide coord order mismatch");
        let (c, mx, _) = cos_stats(&sub.feats, &io["sub_feats"]);
        println!("subdivide cos={c:.8} max|d|={mx:.3e}");
        assert!(c > 0.99999, "cos {c}");
    }
}
