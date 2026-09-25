use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use hanzo_ml::{quantized::GgmlDType, DType, Device, Result, Tensor};
use hanzo_nn::Linear;

mod ops;

#[cfg(feature = "cuda")]
use crate::{
    cublaslt::{maybe_init_cublas_lt_wrapper, supports_f8, CublasLtWrapper, CUBLASLT_CONTROLLER},
    fp8::QuantizationResult,
};
use crate::{
    generate_isq, generate_isq_imatrix,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    make_dummy_or_error,
    utils::{serialize_tensor, UQFF_VERSION},
    AfqBits, AfqGroupSize, AfqLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits, HqqConfig, HqqLayer,
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedConfig, QuantizedSerde,
    QuantizedSerdeType, Shard, ShardedVarBuilder, UnquantLinear,
};

/// FP8 tensor cores read both matrices 16 elements at a time.
#[cfg(feature = "cuda")]
const GEMM_GRANULE: usize = 16;

#[derive(Debug)]
enum Weight {
    Dense(Tensor),
    /// E4M3 [N, K] with the checkpoint's per-tensor dequantization scale. `dtype` is what
    /// activations run in, not what `q` holds.
    #[cfg(feature = "cuda")]
    Packed {
        q: Tensor,
        w_scale: Tensor,
        /// The checkpoint's calibrated activation scale, present only when it quantized
        /// activations too. Activations go against their own amax otherwise.
        x_scale: Option<Tensor>,
        /// cuBLASLt asks for a scale for D as well. The GEMM writes BF16, which is not
        /// requantized, so it is one.
        d_scale: Tensor,
        dtype: DType,
    },
}

/// Per-tensor FP8 linear layer.
///
/// The checkpoint holds an E4M3 weight under one FP32 dequantization scale, and often an
/// activation scale calibrated the same way:
/// - `<layer>.weight` (FP8 E4M3)
/// - `<layer>.weight_scale`, spelled `weight_scale_inv` by some checkpoints
/// - `<layer>.input_scale`, spelled `activation_scale` by some checkpoints, optional
#[derive(Debug)]
pub struct PerTensorFP8Linear {
    weight: Weight,
    bias: Option<Tensor>,
}

/// The weight stays FP8 when cuBLASLt can run the GEMM on it: a device with FP8 tensor
/// cores and a handle, BF16 activations (what the GEMM writes, and so what the bias must
/// be), and a shape those cores can read. Anything else dequantizes at load.
#[cfg(feature = "cuda")]
fn packed_weight(
    weight: &Tensor,
    weight_scale: &Tensor,
    input_scale: Option<&Tensor>,
    bias: Option<&Tensor>,
    dtype: DType,
) -> Result<Option<Weight>> {
    if !supports_f8(weight.device())
        || dtype != DType::BF16
        || bias.is_some_and(|b| b.dtype() != DType::BF16)
        || weight.rank() != 2
        || weight.dims().iter().any(|d| d % GEMM_GRANULE != 0)
    {
        return Ok(None);
    }
    maybe_init_cublas_lt_wrapper(weight.device().clone());
    if CUBLASLT_CONTROLLER
        .get_for_device(weight.device())
        .is_none()
    {
        return Ok(None);
    }

    let scalar = |t: &Tensor| -> Result<Tensor> { t.reshape(())?.to_dtype(DType::F32) };
    // Activations divide by their scale in their own dtype, so the GEMM has to multiply
    // by the scale as that dtype rounds it or the layer picks up a systematic gain.
    let x_scale = input_scale
        .map(|s| -> Result<Tensor> { scalar(s)?.to_dtype(dtype)?.to_dtype(DType::F32) })
        .transpose()?;

    Ok(Some(Weight::Packed {
        q: weight.clone(),
        w_scale: scalar(weight_scale)?,
        x_scale,
        d_scale: Tensor::new(1f32, weight.device())?,
        dtype,
    }))
}

impl QuantMethod for PerTensorFP8Linear {
    fn new(method: QuantMethodConfig) -> hanzo_ml::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::PerTensorFP8 {
                weight,
                weight_scale,
                input_scale,
                bias,
                dequant_dtype,
            } => {
                // Only the tensor core path reads the activation scale.
                #[cfg(not(feature = "cuda"))]
                let _ = input_scale;

                #[cfg(feature = "cuda")]
                if let Some(packed) = packed_weight(
                    &weight,
                    &weight_scale,
                    input_scale.as_ref(),
                    bias.as_ref(),
                    dequant_dtype,
                )? {
                    return Ok(Self {
                        weight: packed,
                        bias,
                    });
                }

                let dense = ops::fp8_pertensor_dequantize(&weight, &weight_scale, dequant_dtype)?;
                Ok(Self {
                    weight: Weight::Dense(dense),
                    bias,
                })
            }
            _ => unreachable!(),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        match &self.weight {
            Weight::Dense(w) => Ok(w.clone()),
            #[cfg(feature = "cuda")]
            Weight::Packed {
                q, w_scale, dtype, ..
            } => ops::fp8_pertensor_dequantize(q, w_scale, *dtype),
        }
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        match &self.weight {
            Weight::Dense(w) => self.dense_matmul(w, x),
            #[cfg(feature = "cuda")]
            Weight::Packed {
                q,
                w_scale,
                x_scale,
                d_scale,
                ..
            } => match CUBLASLT_CONTROLLER.get_for_device(q.device()) {
                Some(handle) => self.fp8_matmul(handle, x, q, w_scale, x_scale.as_ref(), d_scale),
                None => self.dense_matmul(&self.dequantize_w()?, x),
            },
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        hanzo_ml::bail!("PerTensorFP8Linear does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        let device = match &self.weight {
            Weight::Dense(w) => w.device(),
            #[cfg(feature = "cuda")]
            Weight::Packed { q, .. } => q.device(),
        };
        (DType::F8E4M3, device.clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight = self.dequantize_w()?;
        match dtype {
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    hanzo_ml::bail!("HQQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    _ => unreachable!(),
                };
                let cfg = HqqConfig {
                    bits,
                    group_size: ISQ_HQQ_GROUP_SIZE.try_into()?,
                    axis: HqqAxis::Zero,
                    optimization_steps: ISQ_HQQ_DEFAULT_OPT_STEPS,
                    round_zeros: false,
                    channel_wise: true,
                };
                let res = HqqLayer::quantize(&weight.to_device(&device)?, &device, cfg)?;
                if let Some(bias) = &self.bias {
                    let bias = bias
                        .to_device(&device)?
                        .to_dtype(res.dtype_and_device().0)?;
                    Ok(Arc::new(res.with_bias(bias)))
                } else {
                    Ok(Arc::new(res))
                }
            }
            Some(IsqType::AFQ2 | IsqType::AFQ3 | IsqType::AFQ4 | IsqType::AFQ6 | IsqType::AFQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    hanzo_ml::bail!("AFQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::AFQ8 => AfqBits::Eight,
                    IsqType::AFQ6 => AfqBits::Six,
                    IsqType::AFQ4 => AfqBits::Four,
                    IsqType::AFQ3 => AfqBits::Three,
                    IsqType::AFQ2 => AfqBits::Two,
                    _ => unreachable!(),
                };

                Ok(Arc::new(AfqLayer::new(QuantMethodConfig::Afq {
                    weight: weight.to_device(&device)?,
                    bias: self.bias.as_ref().map(|b| b.to_device(&device).unwrap()),
                    bits,
                    group_size: AfqGroupSize::default(),
                })?))
            }
            Some(
                IsqType::Q2K
                | IsqType::Q3K
                | IsqType::Q4K
                | IsqType::Q4_0
                | IsqType::Q4_1
                | IsqType::Q5K
                | IsqType::Q5_0
                | IsqType::Q5_1
                | IsqType::Q6K
                | IsqType::Q8K
                | IsqType::Q8_0
                | IsqType::Q8_1,
            ) => {
                let dtype: GgmlDType = dtype.unwrap().try_into()?;
                let res = if let Some(imatrix_weight) = imatrix_weight {
                    generate_isq_imatrix!(weight, imatrix_weight, device, dtype, n_quantized, guard)
                } else {
                    generate_isq!(weight, device, dtype, n_quantized, guard)
                };
                Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: res,
                    b: self
                        .bias
                        .as_ref()
                        .map(|b| b.to_dtype(DType::F32).unwrap().to_device(&device).unwrap()),
                })?))
            }
            Some(IsqType::F8E4M3) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    hanzo_ml::bail!("F8E4M3 does not support imatrix.");
                }

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(FP8Linear::new(QuantMethodConfig::FP8 {
                    lin: Linear::new(w, b),
                    dtype: DType::F8E4M3,
                })?))
            }
            Some(IsqType::F8Q8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    hanzo_ml::bail!("F8Q8 does not support imatrix.");
                }

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(crate::F8Q8Linear::from_weight(&w, b)?))
            }
            Some(IsqType::MXFP4) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    hanzo_ml::bail!("MXFP4 does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let w = weight.to_device(&device)?;
                let b = self
                    .bias
                    .as_ref()
                    .map(|b| b.to_device(&device))
                    .transpose()?;
                crate::MXFP4Layer::quantize(&w, b, &device)
            }
            None => {
                let _acquired_quantize_guard = guard.acquire(&device);

                let w = weight.to_device(&device)?;
                let b = if let Some(b) = &self.bias {
                    Some(b.to_device(&device)?)
                } else {
                    None
                };
                Ok(Arc::new(UnquantLinear::new(
                    QuantMethodConfig::Unquantized(Linear::new(w, b)),
                )?))
            }
        }
    }
}

impl PerTensorFP8Linear {
    fn dense_matmul(&self, weight: &Tensor, x: &Tensor) -> Result<Tensor> {
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            weight.clone(),
            self.bias.clone(),
        )))?;
        unquant.forward(x)
    }

    /// `x @ qᵀ` on FP8 tensor cores.
    ///
    /// cuBLASLt multiplies each operand by its own dequantization scale inside the GEMM
    /// (`D = alpha * (w_scale * q)ᵀ (x_scale * xq) + beta * C`), so quantized activations
    /// go in and a BF16 result in the checkpoint's units comes back. D carries no scale of
    /// its own because BF16 output is never requantized, and the bias rides the epilogue,
    /// which leaves C out of the sum.
    #[cfg(feature = "cuda")]
    fn fp8_matmul(
        &self,
        handle: &CublasLtWrapper,
        x: &Tensor,
        q: &Tensor,
        w_scale: &Tensor,
        x_scale: Option<&Tensor>,
        d_scale: &Tensor,
    ) -> Result<Tensor> {
        let (out_dim, in_dim) = q.dims2()?;
        let tokens = x.elem_count() / in_dim;
        let mut out_shape = x.dims().to_vec();
        out_shape.pop();
        out_shape.push(out_dim);

        let (xq, dequant_x_scale) = match x_scale {
            Some(scale) => (ops::fp8_pertensor_quantize(x, scale)?, scale.clone()),
            None => {
                let QuantizationResult {
                    qw,
                    dequantize_scale,
                    ..
                } = FP8Linear::quantize(x, DType::F8E4M3)?;
                (qw, dequantize_scale)
            }
        };

        handle
            .batch_matmul_f8(
                &q.unsqueeze(0)?,
                &xq.reshape((1, tokens, in_dim))?,
                w_scale,
                &dequant_x_scale,
                d_scale,
                None,
                None,
                None,
                self.bias.as_ref(),
                None,
            )?
            .reshape(out_shape)
    }
}

// Serialization structure (same as UnquantLinear):
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (1 for unquantized), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for PerTensorFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "pertensor-fp8-linear"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.bias.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        // Serialize as unquantized: the weight is written in its dequantized form.
        let mut buffer = Vec::new();

        // Version is always first!
        buffer.extend(&UQFF_VERSION.to_le_bytes());

        // ISQ type for unquant is 1 (same as UnquantLinear)
        buffer.push(QuantizedSerdeType::Unquant as u8);

        // Has bias
        buffer.push(bias.is_some() as u8);

        // Weight
        serialize_tensor(&mut buffer, &self.dequantize_w()?)?;

        if let Some(bias) = &bias {
            // Bias
            serialize_tensor(&mut buffer, bias)?;
        }

        Ok(Cow::from(buffer))
    }
}

/// Load a per-tensor FP8 linear layer from the VarBuilder.
///
/// This handles models whose FP8 quantization is per-tensor rather than blockwise, so the
/// weight carries one scale, and the activations at most one.
pub fn pertensor_fp8_linear_b(
    in_dim: usize,
    out_dim: usize,
    _config: &QuantizedConfig,
    bias: bool,
    _hints: Shard,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let scale_name = ["weight_scale", "weight_scale_inv"]
        .into_iter()
        .find(|name| vb.contains_tensor(name));

    let Some(scale_name) = scale_name else {
        // Handle the case where we actually have unquantized weights
        if vb.contains_tensor("weight") {
            return crate::linear_b(in_dim, out_dim, bias, &None, vb);
        }
        return make_dummy_or_error("pertensor_fp8_linear", &vb, &["weight", "weight_scale"]);
    };

    // Load FP8 weight tensor
    let weight = vb.get_with_hints_dtype(
        (out_dim, in_dim),
        "weight",
        Default::default(),
        DType::F8E4M3,
    )?;

    // Load per-tensor weight scale (scalar)
    let weight_scale = vb.get_with_hints_dtype((), scale_name, Default::default(), DType::F32)?;

    // Load the activation scale if the checkpoint calibrated one
    let input_scale = ["input_scale", "activation_scale"]
        .into_iter()
        .find(|name| vb.contains_tensor(name))
        .map(|name| vb.get_with_hints_dtype((), name, Default::default(), DType::F32))
        .transpose()?;

    let bias = if bias && vb.contains_tensor("bias") {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };

    // Determine the output dtype for dequantization.
    // We can't use vb.dtype() as that returns F8E4M3 (the storage type).
    // Use the bias dtype if available, otherwise default to BF16.
    let dequant_dtype = bias.as_ref().map(|b| b.dtype()).unwrap_or(DType::BF16);

    Ok(Arc::new(PerTensorFP8Linear::new(
        QuantMethodConfig::PerTensorFP8 {
            weight,
            weight_scale,
            input_scale,
            bias,
            dequant_dtype,
        },
    )?))
}

#[cfg(test)]
mod tests {
    use hanzo_ml::{DType, Device, Result, Tensor};

    use super::{ops::fp8_pertensor_dequantize, PerTensorFP8Linear};
    use crate::{scalar_fp8::ops::dtype_to_fp8, QuantMethod, QuantMethodConfig};

    /// A layer and the exact dequantization of its weight. Both sides of a comparison
    /// then see the same E4M3 grid, so only a wiring error can move the result.
    fn layer(
        device: &Device,
        out_dim: usize,
        in_dim: usize,
        input_scale: Option<Tensor>,
        bias: Option<Tensor>,
    ) -> Result<(PerTensorFP8Linear, Tensor)> {
        let weight_scale = Tensor::new(0.01f32, device)?;
        let weight = dtype_to_fp8(&Tensor::rand(-4f32, 4f32, (out_dim, in_dim), device)?)?;
        let dense = fp8_pertensor_dequantize(&weight, &weight_scale, DType::BF16)?;
        let layer = PerTensorFP8Linear::new(QuantMethodConfig::PerTensorFP8 {
            weight,
            weight_scale,
            input_scale,
            bias,
            dequant_dtype: DType::BF16,
        })?;
        Ok((layer, dense))
    }

    #[test]
    fn cpu_dequantizes_at_load() -> Result<()> {
        let dev = Device::Cpu;
        let (layer, dense) = layer(&dev, 32, 64, None, None)?;

        let got = layer.dequantize_w()?;
        assert_eq!(got.dtype(), DType::BF16);
        assert_eq!(
            got.flatten_all()?.to_vec1::<half::bf16>()?,
            dense.flatten_all()?.to_vec1::<half::bf16>()?
        );

        let x = Tensor::rand(-1f32, 1f32, (8, 64), &dev)?.to_dtype(DType::BF16)?;
        assert_eq!(layer.forward(&x)?.dims(), &[8, 32]);
        Ok(())
    }

    /// The FP8 GEMM against the same matmul run on dequantized operands. Both see the
    /// same quantized activations under a scale BF16 holds exactly, so only the GEMM's
    /// own arithmetic can move the result.
    #[cfg(feature = "cuda")]
    fn gemm_matches_dequantized(
        input_scale: Option<f32>,
        bias: bool,
        x_dims: &[usize],
    ) -> Result<()> {
        use super::ops::fp8_pertensor_quantize;
        use crate::{fp8::QuantizationResult, scalar_fp8::ops::fp8_to_dtype, FP8Linear};

        const IN_DIM: usize = 256;
        const OUT_DIM: usize = 128;

        let Ok(dev) = Device::new_cuda(0) else {
            return Ok(());
        };
        let input_scale = input_scale.map(|s| Tensor::new(s, &dev)).transpose()?;
        let bias = bias
            .then(|| -> Result<Tensor> {
                Tensor::rand(-1f32, 1f32, OUT_DIM, &dev)?.to_dtype(DType::BF16)
            })
            .transpose()?;
        let (layer, dense) = layer(&dev, OUT_DIM, IN_DIM, input_scale.clone(), bias.clone())?;

        let x = Tensor::rand(-1f32, 1f32, x_dims, &dev)?.to_dtype(DType::BF16)?;
        let (xq, x_dequant_scale) = match &input_scale {
            Some(scale) => (fp8_pertensor_quantize(&x, scale)?, scale.clone()),
            None => {
                let QuantizationResult {
                    qw,
                    dequantize_scale,
                    ..
                } = FP8Linear::quantize(&x, DType::F8E4M3)?;
                (qw, dequantize_scale)
            }
        };

        let tokens = x.elem_count() / IN_DIM;
        let want = fp8_to_dtype(&xq, DType::F32)?
            .broadcast_mul(&x_dequant_scale)?
            .to_dtype(DType::BF16)?
            .reshape((tokens, IN_DIM))?
            .matmul(&dense.t()?)?;
        let want = match &bias {
            Some(bias) => want.broadcast_add(bias)?.to_dtype(DType::F32)?,
            None => want.to_dtype(DType::F32)?,
        };

        let out = layer.forward(&x)?;
        assert_eq!(out.dims().last(), Some(&OUT_DIM));
        let got = out.reshape((tokens, OUT_DIM))?.to_dtype(DType::F32)?;

        let error = (&got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        let magnitude = want.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(
            error <= 0.02 * magnitude,
            "off by {error} against {magnitude}"
        );
        Ok(())
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_gemm_static_scale() -> Result<()> {
        gemm_matches_dequantized(Some(0.0625), false, &[4, 16, 256])
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_gemm_dynamic_scale() -> Result<()> {
        gemm_matches_dequantized(None, false, &[64, 256])
    }

    /// One row is what decode asks the GEMM for.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_gemm_single_row() -> Result<()> {
        gemm_matches_dequantized(Some(0.0625), false, &[1, 256])
    }

    /// The bias rides the GEMM's epilogue rather than a second pass.
    #[test]
    #[cfg(feature = "cuda")]
    fn cuda_gemm_bias() -> Result<()> {
        gemm_matches_dequantized(Some(0.0625), true, &[64, 256])
    }
}
