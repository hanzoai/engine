use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::Linear;

#[cfg(all(feature = "cuda", has_nvfp4_cutlass_kernels))]
pub mod cutlass;
#[cfg(feature = "cuda")]
pub(crate) mod ffi;
pub mod ops;

use crate::{
    utils::{serialize_tensor, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
    ShardedVarBuilder, UnquantLinear,
};

pub use ops::{nvfp4_dequantize, FP4_E2M1_LUT, NVFP4_BLOCK_SIZE};

/// A packed row is read by the kernel two blocks at a time, one uint4 per load.
#[cfg(feature = "cuda")]
const KERNEL_K_GRANULE: usize = NVFP4_BLOCK_SIZE * 2;

/// At or below this many rows the matvec kernel wins, so decode stays on it.
#[cfg(feature = "cuda")]
const DECODE_ROWS: usize = 4;

#[derive(Debug)]
enum Weight {
    Dense(Tensor),
    /// A stacked expert bank: E2M1 [E, N, K/2], E4M3 block scales [E, N, K/16] in linear
    /// layout, `global` the experts' FP32 `weight_scale_2` [E], and `alpha` = input_scale x
    /// weight_scale_2 [E]. Activations quantize with `gs` = 1 / input_scale (the largest over
    /// the experts) and multiply in the scaled domain (W4A4).
    Stacked {
        q: Tensor,
        scale: Tensor,
        global: Tensor,
        alpha: Tensor,
        gs: f32,
        dtype: DType,
    },
    /// E2M1 [N, K/2] with fp8 e4m3 per-16 block scales [N, K/16] and the checkpoint's
    /// global FP32 multiplier. `dtype` is what activations run in, not what `q` holds.
    #[cfg(feature = "cuda")]
    Packed {
        q: Tensor,
        scale: Tensor,
        global: f32,
        dtype: DType,
        /// The checkpoint's activation `input_scale`: when present the layer is W4A4 at every M,
        /// quantizing activations as the server does.
        input_scale: Option<f32>,
        /// Present when the device and the checkpoint both allow FP4 activations.
        #[cfg(has_nvfp4_cutlass_kernels)]
        blockscaled: Option<cutlass::Weights>,
    },
}

/// NVIDIA ModelOpt NVFP4 linear layer.
///
/// Packed 4-bit weights (E2M1, 2 values per byte) with block scales
/// (FP8 E4M3, 16 weights per scale) and an optional global FP32 scalar multiplier.
#[derive(Debug)]
pub struct NVFP4Layer {
    weight: Weight,
    bias: Option<Tensor>,
}

#[cfg(feature = "cuda")]
fn global_scale(w_s2: Option<&Tensor>) -> Result<f32> {
    match w_s2 {
        Some(s) => Ok(s.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?[0]),
        None => Ok(1.0),
    }
}

/// Block-scaled weights when the device carries the MMA, the shape meets its
/// alignment, and the checkpoint calibrated an activation scale. Any of those
/// missing leaves the layer on the dequantizing kernel.
#[cfg(all(feature = "cuda", has_nvfp4_cutlass_kernels))]
fn blockscaled_weights(
    weight: &Tensor,
    weight_scale: &Tensor,
    global: f32,
    input_scale: Option<&Tensor>,
    dtype: DType,
) -> Result<Option<cutlass::Weights>> {
    let Device::Cuda(dev) = weight.device() else {
        return Ok(None);
    };
    let Some(input_scale) = input_scale else {
        return Ok(None);
    };
    let n = weight.dims()[0];
    let k = weight.dims()[1] * 2;
    if !matches!(dtype, DType::BF16 | DType::F16)
        || !cutlass::device_supported(dev)
        || !cutlass::shape_supported(n, k)
    {
        return Ok(None);
    }
    let activation_scale = input_scale
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?[0];
    cutlass::Weights::new(weight_scale, global, activation_scale, n, k, dtype).map(Some)
}

impl QuantMethod for NVFP4Layer {
    fn new(method: QuantMethodConfig) -> Result<Self> {
        match method {
            QuantMethodConfig::NVFP4 {
                weight,
                weight_scale,
                weight_scale_2,
                input_scale,
                bias,
                dequant_dtype,
            } => {
                // Only the block-scaled path reads the activation scale.
                #[cfg(not(has_nvfp4_cutlass_kernels))]
                let _ = input_scale;

                // Dequantizing at load inflates a 15 GB checkpoint past what the box holds, and
                // decode then reads bf16 bytes it never needed, so CUDA keeps the weights packed.
                #[cfg(feature = "cuda")]
                if matches!(weight.device(), Device::Cuda(_))
                    && ffi::HAVE_NVFP4_GEMM_KERNELS
                    && weight.rank() == 2
                    && (weight.dims()[1] * 2) % KERNEL_K_GRANULE == 0
                {
                    let global = global_scale(weight_scale_2.as_ref())?;
                    let activation = match input_scale.as_ref() {
                        Some(s) => Some(s.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?[0]),
                        None => None,
                    };
                    #[cfg(has_nvfp4_cutlass_kernels)]
                    let blockscaled = blockscaled_weights(
                        &weight,
                        &weight_scale,
                        global,
                        input_scale.as_ref(),
                        dequant_dtype,
                    )?;
                    return Ok(Self {
                        weight: Weight::Packed {
                            q: weight,
                            scale: weight_scale,
                            global,
                            dtype: dequant_dtype,
                            input_scale: activation,
                            #[cfg(has_nvfp4_cutlass_kernels)]
                            blockscaled,
                        },
                        bias,
                    });
                }

                let dense = ops::nvfp4_dequantize(
                    &weight,
                    &weight_scale,
                    weight_scale_2.as_ref(),
                    dequant_dtype,
                )?;
                Ok(Self {
                    weight: Weight::Dense(dense),
                    bias,
                })
            }
            _ => hanzo_ml::bail!("Invalid QuantMethodConfig for NVFP4Layer"),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        match &self.weight {
            Weight::Dense(w) => Ok(w.clone()),
            Weight::Stacked {
                q,
                scale,
                global,
                dtype,
                ..
            } => ops::nvfp4_dequantize(q, scale, None, DType::F32)?
                .broadcast_mul(&global.reshape(((), 1, 1))?)?
                .to_dtype(*dtype),
            #[cfg(feature = "cuda")]
            Weight::Packed {
                q,
                scale,
                global,
                dtype,
                ..
            } => {
                let s2 = Tensor::new(*global, q.device())?;
                ops::nvfp4_dequantize(q, scale, Some(&s2), *dtype)
            }
        }
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        match &self.weight {
            Weight::Stacked { .. } => {
                hanzo_ml::bail!("a stacked NVFP4 expert bank runs through gather_forward")
            }
            Weight::Dense(w) => {
                let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
                    w.clone(),
                    self.bias.clone(),
                )))?;
                unquant.forward(x)
            }
            #[cfg(feature = "cuda")]
            Weight::Packed {
                q,
                scale,
                global,
                input_scale,
                ..
            } => {
                let dims = x.dims().to_vec();
                let x_2d = if dims.len() > 2 {
                    let tokens: usize = dims[..dims.len() - 1].iter().product();
                    x.reshape((tokens, dims[dims.len() - 1]))?
                } else {
                    x.clone()
                };

                let out = match input_scale {
                    Some(input_scale) => self.w4a4(&x_2d, q, scale, *global, *input_scale)?,
                    None => self.matmul(&x_2d, q, scale, *global)?,
                };

                if dims.len() > 2 {
                    let mut out_dims = dims[..dims.len() - 1].to_vec();
                    out_dims.push(out.dim(1)?);
                    return out.reshape(out_dims);
                }
                Ok(out)
            }
        }
    }

    /// `x` [T, 1|topk, K] (or [T, K]) through the experts `indices` [T, topk] picks: activations
    /// quantized per row with the bank's global scale, W4A4 in the scaled domain, alpha per
    /// expert, [T, topk, N] in the input's dtype.
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let Weight::Stacked {
            q,
            scale,
            alpha,
            gs,
            ..
        } = &self.weight
        else {
            hanzo_ml::bail!("NVFP4 gather_forward needs a stacked expert bank");
        };
        let k = x.dim(hanzo_ml::D::Minus1)?;
        let tokens = indices.dim(0)?;
        let rows = x.elem_count() / k;
        let input_has_topk_dim = rows > tokens;
        let (codes, scales) = crate::quantize::nvfp4(&x.reshape((rows, k))?, *gs)?;
        let y = match x.device() {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => ops::nvfp4_moe_gemm(
                &codes,
                &scales,
                q,
                scale,
                alpha,
                indices,
                input_has_topk_dim,
                x.dtype(),
            )?,
            _ => ops::nvfp4_moe_reference(
                &codes,
                &scales,
                q,
                scale,
                alpha,
                indices,
                input_has_topk_dim,
                x.dtype(),
            )?,
        };
        match &self.bias {
            Some(bias) => y.broadcast_add(bias),
            None => Ok(y),
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        hanzo_ml::bail!("NVFP4Layer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        match &self.weight {
            Weight::Dense(w) => (w.dtype(), w.device().clone()),
            Weight::Stacked { q, dtype, .. } => (*dtype, q.device().clone()),
            #[cfg(feature = "cuda")]
            Weight::Packed { q, dtype, .. } => (*dtype, q.device().clone()),
        }
    }

    fn apply_isq(
        self: Arc<Self>,
        _dtype: Option<IsqType>,
        _device: Device,
        _n_quantized: &AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        Ok(self)
    }
}

impl QuantizedSerde for NVFP4Layer {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "nvfp4-layer"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        let mut buffer = Vec::new();
        buffer.extend(&UQFF_VERSION.to_le_bytes());
        buffer.push(QuantizedSerdeType::Unquant as u8);
        buffer.push(bias.is_some() as u8);
        serialize_tensor(&mut buffer, &self.dequantize_w()?)?;
        if let Some(bias) = &bias {
            serialize_tensor(&mut buffer, bias)?;
        }
        Ok(Cow::from(buffer))
    }
}

impl NVFP4Layer {
    /// Block-scaled tensor cores where the checkpoint and device allow it, the
    /// dequantizing kernel otherwise. Decode stays on the matvec path.
    #[cfg(feature = "cuda")]
    fn matmul(&self, x: &Tensor, q: &Tensor, scale: &Tensor, global: f32) -> Result<Tensor> {
        #[cfg(has_nvfp4_cutlass_kernels)]
        if let Weight::Packed {
            blockscaled: Some(blockscaled),
            ..
        } = &self.weight
        {
            if x.dim(0)? > DECODE_ROWS {
                let out = cutlass::matmul(x, q, blockscaled)?;
                return match &self.bias {
                    Some(bias) => out.broadcast_add(bias),
                    None => Ok(out),
                };
            }
        }
        ops::nvfp4_matmul(x, q, scale, global, self.bias.as_ref())
    }

    /// W4A4 for a dense layer with an activation scale: the served quantizer, then the block-
    /// scaled tensor cores above decode sizes where they exist, else the scaled-domain kernels
    /// with the layer as a bank of one.
    #[cfg(feature = "cuda")]
    fn w4a4(
        &self,
        x: &Tensor,
        q: &Tensor,
        scale: &Tensor,
        global: f32,
        input_scale: f32,
    ) -> Result<Tensor> {
        #[cfg(has_nvfp4_cutlass_kernels)]
        if let Weight::Packed {
            blockscaled: Some(blockscaled),
            ..
        } = &self.weight
        {
            if x.dim(0)? > DECODE_ROWS {
                let out = cutlass::matmul(x, q, blockscaled)?;
                return match &self.bias {
                    Some(bias) => out.broadcast_add(bias),
                    None => Ok(out),
                };
            }
        }
        let m = x.dim(0)?;
        let gs = 1.0 / input_scale;
        let (codes, scales) = crate::quantize::nvfp4(x, gs)?;
        let alpha = Tensor::new(&[input_scale * global], x.device())?;
        let ids = Tensor::zeros((m, 1), DType::U32, x.device())?;
        let out = ops::nvfp4_moe_gemm(
            &codes,
            &scales,
            &q.unsqueeze(0)?,
            &scale.unsqueeze(0)?,
            &alpha,
            &ids,
            false,
            x.dtype(),
        )?
        .squeeze(1)?;
        match &self.bias {
            Some(bias) => out.broadcast_add(bias),
            None => Ok(out),
        }
    }

    /// A stacked bank of `experts` NVFP4 experts, projection `proj` of `vb` = `...experts`:
    /// `{e}.{proj}.{weight, weight_scale, weight_scale_2, input_scale}` for e in 0..experts, each
    /// `[n, k]`. The activation scale is the largest `input_scale` over the experts and over the
    /// projections named in `share` (gate and up quantize one input). Each tensor is gathered on
    /// the host and uploaded once.
    pub fn experts(
        vb: ShardedVarBuilder,
        proj: &str,
        share: &[&str],
        experts: usize,
        n: usize,
        k: usize,
        dtype: DType,
    ) -> Result<Self> {
        let host = vb.clone().set_device(Device::Cpu);
        let dev = vb.device().clone();
        let scalar = |e: usize, p: &str, name: &str| -> Result<f32> {
            host.pp(e)
                .pp(p)
                .get_with_hints_dtype((), name, Default::default(), DType::F32)?
                .to_scalar::<f32>()
        };
        let mut input_scale = 0f32;
        for e in 0..experts {
            for p in share.iter().chain(std::iter::once(&proj)) {
                input_scale = input_scale.max(scalar(e, p, "input_scale")?);
            }
        }
        if input_scale <= 0.0 {
            hanzo_ml::bail!("{}: NVFP4 experts need a positive input_scale", vb.prefix());
        }
        let mut global = Vec::with_capacity(experts);
        let mut alpha = Vec::with_capacity(experts);
        for e in 0..experts {
            let g = scalar(e, proj, "weight_scale_2")?;
            global.push(g);
            alpha.push(input_scale * g);
        }
        // One host buffer per tensor, filled expert by expert, uploaded once and dropped.
        fn bank<T: hanzo_ml::WithDType>(
            host: &ShardedVarBuilder,
            dev: &Device,
            proj: &str,
            name: &str,
            (experts, n, cols): (usize, usize, usize),
            dt: DType,
        ) -> Result<Tensor> {
            let mut buf: Vec<T> = Vec::with_capacity(experts * n * cols);
            for e in 0..experts {
                let t = host.pp(e).pp(proj).get_with_hints_dtype(
                    (n, cols),
                    name,
                    Default::default(),
                    dt,
                )?;
                buf.extend(t.flatten_all()?.to_vec1::<T>()?);
            }
            Tensor::from_vec(buf, (experts, n, cols), &Device::Cpu)?.to_device(dev)
        }
        let q = bank::<u8>(&host, &dev, proj, "weight", (experts, n, k / 2), DType::U8)?;
        let scale = bank::<float8::F8E4M3>(
            &host,
            &dev,
            proj,
            "weight_scale",
            (experts, n, k / NVFP4_BLOCK_SIZE),
            DType::F8E4M3,
        )?;
        Ok(Self {
            weight: Weight::Stacked {
                q,
                scale,
                global: Tensor::from_vec(global, experts, &dev)?,
                alpha: Tensor::from_vec(alpha, experts, &dev)?,
                gs: 1.0 / input_scale,
                dtype,
            },
            bias: None,
        })
    }

    /// Load an NVFP4 linear layer from the VarBuilder.
    pub fn linear_b(
        in_dim: usize,
        out_dim: usize,
        bias: bool,
        vb: ShardedVarBuilder,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight = vb.get_with_hints_dtype(
            (out_dim, in_dim / 2),
            "weight",
            Default::default(),
            DType::U8,
        )?;

        let weight_scale = vb.get_with_hints_dtype(
            (out_dim, in_dim / NVFP4_BLOCK_SIZE),
            "weight_scale",
            Default::default(),
            DType::F8E4M3,
        )?;

        let weight_scale_2 = if vb.contains_tensor("weight_scale_2") {
            Some(vb.get_with_hints_dtype((), "weight_scale_2", Default::default(), DType::F32)?)
        } else {
            None
        };

        let input_scale = if vb.contains_tensor("input_scale") {
            Some(vb.get_with_hints_dtype((), "input_scale", Default::default(), DType::F32)?)
        } else {
            None
        };

        let bias = if bias && vb.contains_tensor("bias") {
            Some(vb.get((out_dim,), "bias")?)
        } else {
            None
        };

        let dequant_dtype = bias.as_ref().map(|b| b.dtype()).unwrap_or(DType::BF16);

        Ok(Arc::new(Self::new(QuantMethodConfig::NVFP4 {
            weight,
            weight_scale,
            weight_scale_2,
            input_scale,
            bias,
            dequant_dtype,
        })?))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use hanzo_ml::{DType, Device, Result, Tensor};

    use super::*;

    /// A random bank: codes over the whole E2M1 range, E4M3 scales around real ones.
    fn bank(e: usize, n: usize, k: usize) -> Result<(Tensor, Tensor, Tensor)> {
        let codes: Vec<u8> = (0..e * n * k / 2)
            .map(|i| ((i * 2654435761usize) >> 7) as u8)
            .collect();
        let q = Tensor::from_vec(codes, (e, n, k / 2), &Device::Cpu)?;
        let s = Tensor::rand(0.002f32, 0.05f32, (e, n, k / 16), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
        let global = Tensor::rand(0.5f32, 2.0f32, e, &Device::Cpu)?;
        Ok((q, s, global))
    }

    fn stacked(q: &Tensor, s: &Tensor, global: &Tensor, input_scale: f32, dev: &Device) -> Result<NVFP4Layer> {
        Ok(NVFP4Layer {
            weight: Weight::Stacked {
                q: q.to_device(dev)?,
                scale: s.to_device(dev)?,
                global: global.to_device(dev)?,
                alpha: (global * input_scale as f64)?.to_device(dev)?,
                gs: 1.0 / input_scale,
                dtype: DType::BF16,
            },
            bias: None,
        })
    }

    fn assert_within_ulp(what: &str, got: &Tensor, want: &Tensor) -> Result<()> {
        let cols = *want.dims().last().unwrap();
        let g = got.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let w = want.to_dtype(DType::F32)?.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(g.len(), w.len(), "{what}");
        for (r, (gr, wr)) in g.chunks(cols).zip(w.chunks(cols)).enumerate() {
            let rms = (wr.iter().map(|v| v * v).sum::<f32>() / cols as f32).sqrt();
            for (i, (a, b)) in gr.iter().zip(wr).enumerate() {
                let bound = f32::from_bits(b.abs().to_bits() & 0x7f80_0000) / 128.0 + rms / 1024.0;
                assert!((a - b).abs() <= bound, "{what}: row {r} col {i}: {a} vs {b}");
            }
        }
        Ok(())
    }

    /// CUDA (vecmat at decode sizes, grouped WMMA above) equals the CPU W4A4 reference.
    #[cfg(feature = "cuda")]
    #[test]
    fn stacked_gather_matches_reference() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let e = 16;
        for (n, k) in [(640, 2560), (2560, 640)] {
            let (q, s, global) = bank(e, n, k)?;
            let input_scale = 0.013f32;
            let gpu = stacked(&q, &s, &global, input_scale, &dev)?;
            let cpu = stacked(&q, &s, &global, input_scale, &Device::Cpu)?;
            for (t, topk) in [(1usize, 1usize), (8, 1), (1, 10), (7, 10), (16, 10), (256, 10)] {
                let ids: Vec<u32> = (0..t * topk).map(|i| ((i * 7 + i / topk) % e) as u32).collect();
                let ids = Tensor::from_vec(ids, (t, topk), &Device::Cpu)?;
                for rows in [1, topk] {
                    let x = (Tensor::randn(0f32, 1f32, (t, rows, k), &Device::Cpu)? * 3.0)?.to_dtype(DType::BF16)?;
                    let want = cpu.gather_forward_raw(&x, &ids)?;
                    let got = gpu.gather_forward_raw(&x.to_device(&dev)?, &ids.to_device(&dev)?)?;
                    assert_eq!(got.dims(), &[t, topk, n]);
                    assert_within_ulp(&format!("({n},{k}) t={t} topk={topk} rows={rows}"), &got, &want)?;
                }
            }
        }
        Ok(())
    }

    /// Bank row e is expert e's bytes; alpha[e] and the activation scale come from the scalars.
    #[test]
    fn experts_loader_stacks_in_order() -> Result<()> {
        let (e, n, k) = (3usize, 16usize, 32usize);
        let mut st = HashMap::new();
        let mut want_q = Vec::new();
        let mut want_s = Vec::new();
        for x in 0..e {
            for proj in ["gate_proj", "up_proj"] {
                let q: Vec<u8> = (0..n * k / 2).map(|i| (i + 31 * x + proj.len()) as u8).collect();
                let s = Tensor::full(0.25f32 * (x + 1) as f32, (n, k / 16), &Device::Cpu)?.to_dtype(DType::F8E4M3)?;
                if proj == "gate_proj" {
                    want_q.push(q.clone());
                    want_s.push(s.clone());
                }
                let p = format!("{x}.{proj}");
                st.insert(format!("{p}.weight"), Tensor::from_vec(q, (n, k / 2), &Device::Cpu)?);
                st.insert(format!("{p}.weight_scale"), s);
                st.insert(format!("{p}.weight_scale_2"), Tensor::new(0.5f32 + x as f32, &Device::Cpu)?);
                let input = if proj == "up_proj" && x == 1 { 0.3f32 } else { 0.1 };
                st.insert(format!("{p}.input_scale"), Tensor::new(input, &Device::Cpu)?);
            }
        }
        let path = std::env::temp_dir().join(format!("hanzo-nvfp4-experts-{}.safetensors", std::process::id()));
        hanzo_ml::safetensors::save(&st, &path)?;
        let vb = unsafe {
            crate::ShardedSafeTensors::sharded(&[&path], DType::BF16, &Device::Cpu, None, Arc::new(|_| true))?
        };
        let layer = NVFP4Layer::experts(vb, "gate_proj", &["up_proj"], e, n, k, DType::BF16)?;
        let Weight::Stacked { q, scale, global, alpha, gs, .. } = &layer.weight else {
            panic!("not a stacked bank")
        };
        let qv = q.to_vec3::<u8>()?;
        for x in 0..e {
            let flat: Vec<u8> = qv[x].iter().flatten().copied().collect();
            assert_eq!(flat, want_q[x], "expert {x} codes");
            assert_eq!(
                scale.get(x)?.to_dtype(DType::F32)?.to_vec2::<f32>()?,
                want_s[x].to_dtype(DType::F32)?.to_vec2::<f32>()?
            );
        }
        // up's 0.3 is the largest activation scale over gate and up
        assert_eq!(*gs, 1.0 / 0.3f32);
        assert_eq!(global.to_vec1::<f32>()?, [0.5, 1.5, 2.5]);
        assert_eq!(alpha.to_vec1::<f32>()?, [0.3 * 0.5f32, 0.3 * 1.5, 0.3 * 2.5]);
        Ok(())
    }

    /// A dense layer with an activation scale is W4A4 at every M: CUDA equals the CPU reference
    /// (quantize with the served formula, exact dequant, f32 dot, alpha).
    #[cfg(feature = "cuda")]
    #[test]
    fn dense_w4a4_matches_reference() -> Result<()> {
        let Ok(dev) = Device::new_cuda(0) else { return Ok(()) };
        let (n, k) = (256usize, 1024usize);
        let (q, s, _) = bank(1, n, k)?;
        let (q, s) = (q.squeeze(0)?, s.squeeze(0)?);
        let (global, input_scale) = (0.75f32, 0.021f32);
        let layer = NVFP4Layer::new(QuantMethodConfig::NVFP4 {
            weight: q.to_device(&dev)?,
            weight_scale: s.to_device(&dev)?,
            weight_scale_2: Some(Tensor::new(global, &dev)?),
            input_scale: Some(Tensor::new(input_scale, &dev)?),
            bias: None,
            dequant_dtype: DType::BF16,
        })?;
        let cpu = stacked(&q.unsqueeze(0)?, &s.unsqueeze(0)?, &Tensor::new(&[global], &Device::Cpu)?, input_scale, &Device::Cpu)?;
        for m in [1usize, 3, 4, 5, 64] {
            let x = (Tensor::randn(0f32, 1f32, (m, k), &Device::Cpu)? * 2.0)?.to_dtype(DType::BF16)?;
            let got = layer.forward_raw(&x.to_device(&dev)?)?;
            let want = cpu
                .gather_forward_raw(&x, &Tensor::zeros((m, 1), DType::U32, &Device::Cpu)?)?
                .squeeze(1)?;
            // the block-scaled tensor cores add one more summation order past decode sizes
            assert_within_ulp(&format!("m={m}"), &got, &want)?;
        }
        Ok(())
    }
}
