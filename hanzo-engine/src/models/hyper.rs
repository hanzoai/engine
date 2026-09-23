#![allow(clippy::cast_precision_loss)]
// The first caller is the qwen4exp loader.
#![allow(dead_code)]

//! Gated-residual hyper-connections, the residual of Qwen3.8-Flash-Next (`qwen4exp`).
//!
//! The residual is `n` parallel streams. Each block reads one gated mean of them and writes its
//! output back into every stream through that stream's own gate (vLLM
//! `qwen4_exp/common/hyperconnection.py:196-242`, served as the kernels in `nvidia/ops/hc.py`):
//!
//! ```text
//! xn_s = x_s / rms(x_s) · w_s                                  per stream, w = 1 + w_gemma
//! u    = (1/n) Σ_s σ(W_up silu(W_down vec(xn) / n))_s ⊙ xn_s   the block input
//! x_s ← x_s + 2σ((W_inj vec(xn))_s / n) · f(u)
//! ```
//!
//! The mixer before the LM head is the same `u` with no write-back. States are
//! `[batch, seq, n, hidden]`, stream outer, which is vLLM's `[..., n·hidden]` layout.
//!
//! The norm, `silu(r / n)`, the gated mean and the write-back each run in f32 and round once to
//! the activation dtype (`ops/hc.py:42-52, 100-105, 150-160, 224-229`). The projections return
//! the activation dtype, as vLLM's bf16 GEMMs do.
//!
//! GGUF, per `prefix` `blk.{i}.hc_attn`, `blk.{i}.hc_ffn` or the head's `output_hc`, as loaded
//! (`[out, in]`; the header lists them reversed): `{prefix}_norm.weight` `[n·h]` F32 with the
//! Gemma `+1` folded in, `{prefix}_down.weight` `[rank, n·h]`, `{prefix}_up.weight` `[n·h, rank]`
//! and, on the blocks only, `{prefix}_inject.weight` `[n, n·h]` F32.

use std::io::{Read, Seek};
use std::sync::Arc;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_quant::QuantMethod;

use crate::gguf::Content;
use crate::models::gdn::sigmoid;
use crate::models::quantized_qwen3_5_moe::gguf_qmm;

/// Reads the streams into one block input.
pub(crate) struct Mixer {
    /// Per-stream norm weight, `[n, hidden]` f32.
    norm: Tensor,
    /// Ones, `[hidden]`: the fused RMS norm's weight, as `norm` differs per stream.
    unit: Tensor,
    down: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    eps: f32,
}

impl Mixer {
    /// The mixer at `prefix` over `n` streams.
    pub(crate) fn from_gguf<R: Read + Seek>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        n: usize,
        eps: f32,
        dev: &Device,
    ) -> Result<Self> {
        let norm = ct
            .tensor(&format!("{prefix}_norm.weight"), dev)?
            .dequantize(dev)?
            .reshape((n, ()))?;
        Ok(Self {
            unit: Tensor::ones(norm.dim(1)?, DType::F32, dev)?,
            norm,
            down: gguf_qmm(ct.tensor(&format!("{prefix}_down.weight"), dev)?)?,
            up: gguf_qmm(ct.tensor(&format!("{prefix}_up.weight"), dev)?)?,
            eps,
        })
    }

    /// `xn_s = x_s / rms(x_s) · w_s` for each stream of `[b, s, n, h]` (`ops/hc.py:42-52`).
    pub(crate) fn norm(&self, x: &Tensor) -> Result<Tensor> {
        let x32 = x.to_dtype(DType::F32)?.contiguous()?;
        hanzo_nn::ops::rms_norm(&x32, &self.unit, self.eps)?
            .broadcast_mul(&self.norm)?
            .to_dtype(x.dtype())
    }

    /// The block input `[b, s, h]` from the normed streams (`nvidia/hyperconnection.py:137-148`).
    pub(crate) fn mix(&self, xn: &Tensor) -> Result<Tensor> {
        let (b, s, n, h) = xn.dims4()?;
        let r = self.down.forward(&xn.reshape((b, s, n * h))?)?;
        let g = self.up.forward(&silu(&r, n)?)?;
        mean(xn, &g.reshape((b, s, n, h))?)
    }
}

/// A block inside a hyper-connection: its input comes from the [`Mixer`] and its output goes
/// back into every stream.
pub(crate) struct Branch {
    mixer: Mixer,
    inject: Arc<dyn QuantMethod>,
}

impl Branch {
    /// The branch at `prefix` over `n` streams.
    pub(crate) fn from_gguf<R: Read + Seek>(
        ct: &mut Content<'_, R>,
        prefix: &str,
        n: usize,
        eps: f32,
        dev: &Device,
    ) -> Result<Self> {
        Ok(Self {
            mixer: Mixer::from_gguf(ct, prefix, n, eps, dev)?,
            inject: gguf_qmm(ct.tensor(&format!("{prefix}_inject.weight"), dev)?)?,
        })
    }

    /// `x_s + 2σ((W_inj vec(xn))_s / n) · f(u)` over `[b, s, n, h]`, where `f` maps the block
    /// input `[b, s, h]` to the block output (`common/hyperconnection.py:203-242`).
    pub(crate) fn apply(
        &self,
        x: &Tensor,
        f: impl FnOnce(&Tensor) -> Result<Tensor>,
    ) -> Result<Tensor> {
        let (b, s, n, h) = x.dims4()?;
        let xn = self.mixer.norm(x)?;
        // vLLM takes these logits from the down projection's GEMM, in the activation dtype
        // (`nvidia/hyperconnection.py:139-141`).
        let inj = self.inject.forward(&xn.reshape((b, s, n * h))?)?;
        let y = f(&self.mixer.mix(&xn)?)?;
        combine(x, &y, &inj)
    }
}

/// The streams at the embedding: `n` copies of `e`, `[b, s, h] -> [b, s, n, h]`
/// (`nvidia/model.py:489`).
pub(crate) fn expand(e: &Tensor, n: usize) -> Result<Tensor> {
    let (b, s, h) = e.dims3()?;
    e.unsqueeze(2)?.broadcast_as((b, s, n, h))?.contiguous()
}

/// `silu(r / n)` in f32, rounded once (`ops/hc.py:100-101`).
fn silu(r: &Tensor, n: usize) -> Result<Tensor> {
    (r.to_dtype(DType::F32)? / n as f64)?
        .silu()?
        .to_dtype(r.dtype())
}

/// `(1/n) Σ_s σ(g_s) ⊙ xn_s` over `[b, s, n, h]`, accumulated in f32 and rounded once
/// (`ops/hc.py:150-156`).
fn mean(xn: &Tensor, g: &Tensor) -> Result<Tensor> {
    let n = xn.dim(2)?;
    let gated = (sigmoid(&g.to_dtype(DType::F32)?)? * xn.to_dtype(DType::F32)?)?;
    (gated.sum(2)? / n as f64)?.to_dtype(xn.dtype())
}

/// `x_s + 2σ(inj_s / n) · y` for `x` `[b, s, n, h]`, `y` `[b, s, h]` and `inj` `[b, s, n]`, in
/// f32 and rounded once (`ops/hc.py:224-225`).
fn combine(x: &Tensor, y: &Tensor, inj: &Tensor) -> Result<Tensor> {
    let n = x.dim(2)?;
    let gate = (sigmoid(&(inj.to_dtype(DType::F32)? / n as f64)?)? * 2.0)?;
    let y = y
        .to_dtype(DType::F32)?
        .unsqueeze(2)?
        .broadcast_mul(&gate.unsqueeze(3)?)?;
    x.to_dtype(DType::F32)?
        .broadcast_add(&y)?
        .to_dtype(x.dtype())
}

/// vLLM's `GatedResidual` in f64 over one token's `[n·h]` row, stream outer
/// (`common/hyperconnection.py`, `nvidia/ops/hc.py`).
#[cfg(test)]
mod reference {
    pub(super) fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// `F.linear(x, w)` for a row-major `[out, in]` weight.
    pub(super) fn linear(w: &[f64], x: &[f64]) -> Vec<f64> {
        w.chunks(x.len())
            .map(|row| row.iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// `GroupedGemmaRMSNorm` with one group per stream (`common/hyperconnection.py:80-87`,
    /// `ops/hc.py:45-48`). The GGUF folds `1 + w` into `w`, so `w` multiplies directly.
    pub(super) fn norm(x: &[f64], w: &[f64], h: usize, eps: f64) -> Vec<f64> {
        let mut out = Vec::with_capacity(x.len());
        for (xs, ws) in x.chunks(h).zip(w.chunks(h)) {
            let rrms = 1.0 / (xs.iter().map(|v| v * v).sum::<f64>() / h as f64 + eps).sqrt();
            out.extend(xs.iter().zip(ws).map(|(v, w)| v * rrms * w));
        }
        out
    }

    /// `silu(r / n)` (`common/hyperconnection.py:212-215`, `ops/hc.py:100-101`).
    pub(super) fn silu(r: f64, n: usize) -> f64 {
        let x = r / n as f64;
        x * sigmoid(x)
    }

    /// `(1/n) Σ_s σ(g_s) xn_s` (`common/hyperconnection.py:216-221`, `ops/hc.py:150-156`).
    pub(super) fn mean(xn: &[f64], g: &[f64], h: usize) -> Vec<f64> {
        let n = xn.len() / h;
        (0..h)
            .map(|i| {
                (0..n)
                    .map(|s| sigmoid(g[s * h + i]) * xn[s * h + i])
                    .sum::<f64>()
                    / n as f64
            })
            .collect()
    }

    /// The block input from the normed streams (`common/hyperconnection.py:210-221`).
    pub(super) fn mix(xn: &[f64], down: &[f64], up: &[f64], h: usize) -> Vec<f64> {
        let n = xn.len() / h;
        let lora: Vec<f64> = linear(down, xn).into_iter().map(|r| silu(r, n)).collect();
        mean(xn, &linear(up, &lora), h)
    }

    /// `x_s + 2σ(inj_s / n) · y` (`common/hyperconnection.py:237-241`, `ops/hc.py:224-225`).
    pub(super) fn combine(x: &[f64], y: &[f64], inj: &[f64]) -> Vec<f64> {
        let n = inj.len();
        let mut out = Vec::with_capacity(x.len());
        for (xs, i) in x.chunks(y.len()).zip(inj) {
            let gate = 2.0 * sigmoid(i / n as f64);
            out.extend(xs.iter().zip(y).map(|(x, y)| x + gate * y));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::quantized::{gguf_file, GgmlDType, QTensor};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    const N: usize = 4;
    const H: usize = 64;
    const RANK: usize = 8;
    const EPS: f32 = 1e-6;
    const BATCH: usize = 2;
    const SEQ: usize = 3;

    /// One hyper-connection's weights, row-major `[out, in]`.
    struct Weights {
        norm: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Vec<f32>,
    }

    fn uniform(rng: &mut StdRng, len: usize, lo: f32, hi: f32) -> Vec<f32> {
        (0..len).map(|_| rng.random_range(lo..hi)).collect()
    }

    /// Scales that keep the gates away from saturation, so every term reaches the output.
    fn weights(rng: &mut StdRng) -> Weights {
        Weights {
            norm: uniform(rng, N * H, 0.6, 1.2),
            down: uniform(rng, RANK * N * H, -0.15, 0.15),
            up: uniform(rng, N * H * RANK, -1.5, 1.5),
            inject: uniform(rng, N * N * H, -0.2, 0.2),
        }
    }

    fn tensor(data: &[f32], dims: &[usize]) -> Result<Tensor> {
        Tensor::from_vec(data.to_vec(), dims, &Device::Cpu)
    }

    fn f64s(t: &Tensor) -> Result<Vec<f64>> {
        Ok(t.to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f64::from)
            .collect())
    }

    fn widen(v: &[f32]) -> Vec<f64> {
        v.iter().copied().map(f64::from).collect()
    }

    /// A block branch at `blk.0.hc_attn` and the head mixer at `output_hc`, read back from an
    /// unquantized GGUF with the real file's names and dims.
    fn load(block: &Weights, head: &Weights) -> Result<(Branch, Mixer)> {
        let dev = Device::Cpu;
        let mut owned = Vec::new();
        for (prefix, w) in [("blk.0.hc_attn", block), ("output_hc", head)] {
            let mut put = |name: &str, data: &[f32], dims: &[usize]| -> Result<()> {
                owned.push((
                    format!("{prefix}_{name}.weight"),
                    QTensor::quantize(&tensor(data, dims)?, GgmlDType::F32)?,
                ));
                Ok(())
            };
            put("norm", &w.norm, &[N * H])?;
            put("down", &w.down, &[RANK, N * H])?;
            put("up", &w.up, &[N * H, RANK])?;
            if prefix != "output_hc" {
                put("inject", &w.inject, &[N, N * H])?;
            }
        }
        let tensors: Vec<(&str, &QTensor)> = owned.iter().map(|(n, t)| (n.as_str(), t)).collect();
        let dir = tempfile::tempdir().map_err(hanzo_ml::Error::msg)?;
        let path = dir.path().join("hyper.gguf");
        // Content wants an architecture it knows; the loaders read only the tensors above.
        let arch = gguf_file::Value::String("qwen35moe".to_string());
        let mut file = std::fs::File::create(&path).map_err(hanzo_ml::Error::msg)?;
        gguf_file::write(&mut file, &[("general.architecture", &arch)], &tensors)?;

        let mut files = [std::fs::File::open(&path).map_err(hanzo_ml::Error::msg)?];
        let mut readers: Vec<&mut std::fs::File> = files.iter_mut().collect();
        let mut ct = Content::from_readers(&mut readers)?;
        Ok((
            Branch::from_gguf(&mut ct, "blk.0.hc_attn", N, EPS, &dev)?,
            Mixer::from_gguf(&mut ct, "output_hc", N, EPS, &dev)?,
        ))
    }

    /// Largest deviation from the reference over its largest magnitude.
    fn relative(got: &[f64], want: &[f64]) -> f64 {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        let worst = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (g, w)| m.max((g - w).abs()));
        worst / scale
    }

    fn check_mix(mixer: &Mixer, w: &Weights, rng: &mut StdRng) -> Result<()> {
        let x = uniform(rng, BATCH * SEQ * N * H, -1.0, 1.0);
        let got = f64s(&mixer.mix(&mixer.norm(&tensor(&x, &[BATCH, SEQ, N, H])?)?)?)?;
        let (norm, down, up) = (widen(&w.norm), widen(&w.down), widen(&w.up));
        let want: Vec<f64> = widen(&x)
            .chunks(N * H)
            .flat_map(|row| {
                reference::mix(
                    &reference::norm(row, &norm, H, f64::from(EPS)),
                    &down,
                    &up,
                    H,
                )
            })
            .collect();
        let err = relative(&got, &want);
        assert!(err < 1e-5, "mix is {err:.2e} off the reference");
        Ok(())
    }

    #[test]
    fn mix_matches_reference() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(0x6879);
        let (block, head) = (weights(&mut rng), weights(&mut rng));
        let (branch, _) = load(&block, &head)?;
        check_mix(&branch.mixer, &block, &mut rng)
    }

    /// The head mixer reads the same three tensors under `output_hc` and has no `_inject`.
    #[test]
    fn head_matches_reference() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(0x6865);
        let (block, head) = (weights(&mut rng), weights(&mut rng));
        let (_, mixer) = load(&block, &head)?;
        check_mix(&mixer, &head, &mut rng)
    }

    /// A branch around a linear block equals the reference's mix, block and combine.
    #[test]
    fn apply_matches_reference() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(0x6170);
        let (block, head) = (weights(&mut rng), weights(&mut rng));
        let (branch, _) = load(&block, &head)?;
        let f = uniform(&mut rng, H * H, -0.2, 0.2);
        let x = uniform(&mut rng, BATCH * SEQ * N * H, -1.0, 1.0);

        let ft = tensor(&f, &[H, H])?;
        let got = f64s(&branch.apply(&tensor(&x, &[BATCH, SEQ, N, H])?, |u| {
            u.broadcast_matmul(&ft.t()?)
        })?)?;

        let (norm, down, up, inject) = (
            widen(&block.norm),
            widen(&block.down),
            widen(&block.up),
            widen(&block.inject),
        );
        let f = widen(&f);
        let want: Vec<f64> = widen(&x)
            .chunks(N * H)
            .flat_map(|row| {
                let xn = reference::norm(row, &norm, H, f64::from(EPS));
                let y = reference::linear(&f, &reference::mix(&xn, &down, &up, H));
                reference::combine(row, &y, &reference::linear(&inject, &xn))
            })
            .collect();
        let err = relative(&got, &want);
        assert!(err < 1e-5, "apply is {err:.2e} off the reference");
        Ok(())
    }

    /// A block that writes nothing leaves every stream bit-for-bit as it was.
    #[test]
    fn apply_with_zero_block_is_identity() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(0x7a65);
        let (block, head) = (weights(&mut rng), weights(&mut rng));
        let (branch, _) = load(&block, &head)?;
        let x = tensor(
            &uniform(&mut rng, BATCH * SEQ * N * H, -1.0, 1.0),
            &[BATCH, SEQ, N, H],
        )?;
        for dtype in [DType::F32, DType::BF16] {
            let x = x.to_dtype(dtype)?;
            let out = branch.apply(&x, |u| u.zeros_like())?;
            assert_eq!(out.dtype(), dtype);
            let bits = |t: &Tensor| -> Result<Vec<u32>> {
                Ok(t.to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?
                    .into_iter()
                    .map(f32::to_bits)
                    .collect())
            };
            assert_eq!(bits(&out)?, bits(&x)?, "{dtype:?} streams moved");
        }
        Ok(())
    }

    /// Half the spacing of bf16 values around `v`, the most one rounding can move it: bf16 keeps
    /// 8 significant bits, so it is `2^(⌊log2 |v|⌋ - 8)`.
    fn half_ulp(v: f64) -> f64 {
        f64::from_bits(v.abs().to_bits() & 0x7ff0_0000_0000_0000) / 256.0
    }

    /// Asserts each bf16 value is the exact result rounded once: within half a bf16 ulp of the
    /// f64 reference, plus the f32 arithmetic's own error, well under an ulp. A second bf16
    /// rounding inside the step would cost up to another half ulp.
    fn assert_rounded_once(what: &str, got: &[f64], want: &[f64]) {
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs()));
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let err = (g - w).abs();
            assert!(
                err <= half_ulp(*w) + 1e-6 * scale,
                "{what}[{i}] = {g}, reference {w}: {err:.3e} is more than one rounding"
            );
        }
    }

    /// `silu(r / n)`, the gated mean and the write-back each keep f32 until one final rounding to
    /// bf16, as vLLM's kernels do. The bound is half a bf16 ulp rather than 1e-5 because the bf16
    /// rounding is what is under test.
    #[test]
    fn f32_steps_round_once() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(0x726f);
        let mut bf16 = |len: usize, lo: f32, hi: f32, dims: &[usize]| -> Result<Tensor> {
            tensor(&uniform(&mut rng, len, lo, hi), dims)?.to_dtype(DType::BF16)
        };
        let r = bf16(BATCH * SEQ * RANK, -6.0, 6.0, &[BATCH, SEQ, RANK])?;
        let xn = bf16(BATCH * SEQ * N * H, -2.0, 2.0, &[BATCH, SEQ, N, H])?;
        let g = bf16(BATCH * SEQ * N * H, -3.0, 3.0, &[BATCH, SEQ, N, H])?;
        let x = bf16(BATCH * SEQ * N * H, -2.0, 2.0, &[BATCH, SEQ, N, H])?;
        let y = bf16(BATCH * SEQ * H, -2.0, 2.0, &[BATCH, SEQ, H])?;
        let inj = bf16(BATCH * SEQ * N, -8.0, 8.0, &[BATCH, SEQ, N])?;

        let want: Vec<f64> = f64s(&r)?
            .into_iter()
            .map(|v| reference::silu(v, N))
            .collect();
        assert_rounded_once("silu", &f64s(&silu(&r, N)?)?, &want);

        let (xn64, g64) = (f64s(&xn)?, f64s(&g)?);
        let want: Vec<f64> = xn64
            .chunks(N * H)
            .zip(g64.chunks(N * H))
            .flat_map(|(xn, g)| reference::mean(xn, g, H))
            .collect();
        assert_rounded_once("mean", &f64s(&mean(&xn, &g)?)?, &want);

        let (x64, y64, inj64) = (f64s(&x)?, f64s(&y)?, f64s(&inj)?);
        let want: Vec<f64> = x64
            .chunks(N * H)
            .zip(y64.chunks(H))
            .zip(inj64.chunks(N))
            .flat_map(|((x, y), inj)| reference::combine(x, y, inj))
            .collect();
        assert_rounded_once("combine", &f64s(&combine(&x, &y, &inj)?)?, &want);
        Ok(())
    }

    #[test]
    fn expand_repeats_the_embedding() -> Result<()> {
        let e = Tensor::arange(0f32, (BATCH * SEQ * H) as f32, &Device::Cpu)?
            .reshape((BATCH, SEQ, H))?;
        let x = expand(&e, N)?;
        assert_eq!(x.dims(), &[BATCH, SEQ, N, H]);
        for s in 0..N {
            assert_eq!(
                x.narrow(2, s, 1)?.squeeze(2)?.to_vec3::<f32>()?,
                e.to_vec3::<f32>()?
            );
        }
        Ok(())
    }
}
