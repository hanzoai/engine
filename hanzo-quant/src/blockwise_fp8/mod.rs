use std::sync::{atomic::AtomicUsize, Arc};

use hanzo_ml::{quantized::GgmlDType, DType, Device, Result, Tensor};
use hanzo_nn::Linear;

mod ops;
pub use ops::{fp8_blockwise_dequantize, fp8_blockwise_quantize};
#[cfg(feature = "cuda")]
#[allow(unused_imports)]
pub(crate) use ops::{fp8_blockwise_matmul, fp8_indexed_moe_gemm};

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(has_blockwise_fp8_cutlass_kernels)]
pub mod cutlass;
#[cfg(test)]
mod fixture;

use crate::{
    generate_isq, generate_isq_imatrix, has_missing_required_tensors,
    hqq::{ISQ_HQQ_DEFAULT_OPT_STEPS, ISQ_HQQ_GROUP_SIZE},
    make_dummy_or_error, AfqBits, AfqGroupSize, AfqLayer, FP8Linear, GgufMatMul, HqqAxis, HqqBits,
    HqqConfig, HqqLayer, IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard,
    QuantizedConfig, QuantizedSerde, Shard, ShardedVarBuilder, UnquantLinear,
};

#[derive(Debug)]
pub struct BlockwiseFP8Linear {
    weight: Tensor,
    weight_scale_inv: Tensor,
    bias: Option<Tensor>,
    dequant_dtype: DType,
    weight_block_size: Vec<usize>,
    /// The served activation quantizer this layer's input takes.
    act: crate::quantize::Fp8Mode,
}

impl QuantMethod for BlockwiseFP8Linear {
    fn new(method: QuantMethodConfig) -> hanzo_ml::Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { .. }
            | QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. }
            | QuantMethodConfig::NVFP4 { .. } => unreachable!(),
            QuantMethodConfig::BlockwiseFP8 {
                weight,
                weight_scale_inv,
                bias,
                dequant_dtype,
                weight_block_size,
            } => Ok(Self {
                weight,
                weight_scale_inv,
                bias,
                dequant_dtype,
                weight_block_size,
                act: crate::quantize::Fp8Mode::Linear,
            }),
        }
    }
    fn dequantize_w(&self) -> Result<hanzo_ml::Tensor> {
        ops::fp8_blockwise_dequantize(
            &self.weight,
            &self.weight_scale_inv,
            self.weight_block_size.to_vec(),
            self.dequant_dtype,
        )
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        // Try to use native FP8 GEMM kernel on CUDA
        #[cfg(feature = "cuda")]
        {
            if matches!(x.device(), hanzo_ml::Device::Cuda(_)) && ffi::HAVE_BLOCKWISE_GEMM_KERNELS {
                // Handle batched inputs by flattening to 2D
                let orig_dims = x.dims().to_vec();
                let x_2d = if orig_dims.len() > 2 {
                    // Flatten all but last dim: [batch, seq, features] -> [batch*seq, features]
                    let features = orig_dims[orig_dims.len() - 1];
                    let batch_size: usize = orig_dims[..orig_dims.len() - 1].iter().product();
                    x.reshape((batch_size, features))?
                } else {
                    x.clone()
                };

                // Use native FP8 GEMM kernel
                let result = ops::fp8_blockwise_matmul(
                    &x_2d,
                    &self.weight,
                    &self.weight_scale_inv,
                    &self.weight_block_size,
                    self.act,
                )?;

                // Reshape back to original batch dimensions
                let result = if orig_dims.len() > 2 {
                    let out_features = result.dim(1)?;
                    let mut new_dims = orig_dims[..orig_dims.len() - 1].to_vec();
                    new_dims.push(out_features);
                    result.reshape(new_dims)?
                } else {
                    result
                };

                // Apply bias if present
                if let Some(ref bias) = self.bias {
                    return result.broadcast_add(bias);
                }
                return Ok(result);
            }
        }

        // CPU: the same W8A8 math with IEEE arithmetic, both operands dequantized exactly to f32.
        let y = ops::w8a8_reference(
            x,
            &self.weight,
            &self.weight_scale_inv,
            &self.weight_block_size,
            self.act,
        )?;
        match &self.bias {
            Some(bias) => y.broadcast_add(bias),
            None => Ok(y),
        }
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, 1, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts_per_tok).
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // Try to use native FP8 indexed MoE GEMM kernel on CUDA
        #[cfg(feature = "cuda")]
        {
            if matches!(x.device(), hanzo_ml::Device::Cuda(_)) && ffi::HAVE_BLOCKWISE_GEMM_KERNELS {
                // Use native FP8 indexed MoE GEMM kernel (expects U32 indices)
                let result = ops::fp8_indexed_moe_gemm(
                    x,
                    &self.weight,
                    &self.weight_scale_inv,
                    indices,
                    &self.weight_block_size,
                )?;
                // Apply bias if present (broadcast over tokens and topk)
                if let Some(ref bias) = self.bias {
                    return result.broadcast_add(bias);
                }
                return Ok(result);
            }
        }

        // CPU: quantize each input row as the MoE gather does, dequantize both operands exactly
        // to f32 and multiply per (token, slot) by its expert.
        let (n_tokens, topk) = indices.dims2()?;
        let (n_experts, out_features, k) = self.weight.dims3()?;
        let rows = x.elem_count() / k;
        let (qa, sa) = crate::quantize::fp8(&x.reshape((rows, k))?, crate::quantize::Fp8Mode::Gather)?;
        let a = qa
            .to_dtype(DType::F32)?
            .reshape((rows, k / 128, 128))?
            .broadcast_mul(&sa.unsqueeze(2)?)?
            .reshape((n_tokens, rows / n_tokens, k))?;
        let experts = (0..n_experts)
            .map(|e| {
                ops::fp8_blockwise_dequantize(
                    &self.weight.get(e)?,
                    &self.weight_scale_inv.get(e)?,
                    self.weight_block_size.clone(),
                    DType::F32,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let ids = indices.to_dtype(DType::U32)?.to_vec2::<u32>()?;
        let mut out = Vec::with_capacity(n_tokens * topk);
        for (t, row) in ids.iter().enumerate() {
            for (j, &e) in row.iter().enumerate() {
                let r = if a.dim(1)? > 1 { j } else { 0 };
                let xr = a.get(t)?.get(r)?.unsqueeze(0)?;
                out.push(xr.matmul(&experts[e as usize].t()?)?.squeeze(0)?);
            }
        }
        let result = Tensor::stack(&out, 0)?
            .reshape((n_tokens, topk, out_features))?
            .to_dtype(x.dtype())?;
        match &self.bias {
            Some(bias) => result.broadcast_add(bias),
            None => Ok(result),
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        hanzo_ml::bail!("BlockwiseFP8Linear does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, hanzo_ml::Device) {
        (DType::F8E4M3, self.weight.device().clone())
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let weight = ops::fp8_blockwise_dequantize(
            &self.weight,
            &self.weight_scale_inv,
            self.weight_block_size.to_vec(),
            self.dequant_dtype,
        )?;
        match dtype {
            /*Some(IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            Some(IsqType::HQQ4 | IsqType::HQQ8) => {
                let _acquired_quantize_guard = guard.acquire(&device);
                if imatrix_weight.is_some() {
                    // TODO just warn?
                    hanzo_ml::bail!("HQQ does not support imatrix.");
                }

                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let bits = match dtype.unwrap() {
                    IsqType::HQQ8 => HqqBits::Eight,
                    IsqType::HQQ4 => HqqBits::Four,
                    // IsqType::HQQ3 => HqqBits::Three,
                    // IsqType::HQQ2 => HqqBits::Two,
                    // IsqType::HQQ1 => HqqBits::One,
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
                    // TODO just warn?
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
                    // TODO just warn?
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
                // Ignore imatrix altogether

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

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (3 for fp8), u8, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Weight tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------
// Dequant W scalar, f32, little endian
// -----------------------
// Dequant X scalar, f32, little endian
// -----------------------
// Quant scalar, f32, little endian
// -----------------------
// Quantization type, u32, little endian
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for BlockwiseFP8Linear {
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "blockwise-fp8-linear"
    }
}

/// Create a BlockwiseFP8Linear for MoE with 3D weights [num_experts, N, K].
/// This is used by FusedExperts to enable gather_forward with native FP8 GEMM.
pub fn blockwise_fp8_moe(
    weight: Tensor,
    weight_scale_inv: Tensor,
    weight_block_size: Vec<usize>,
    dequant_dtype: DType,
) -> Result<Arc<dyn QuantMethod>> {
    blockwise_fp8_act(
        weight,
        weight_scale_inv,
        weight_block_size,
        dequant_dtype,
        crate::quantize::Fp8Mode::Linear,
    )
}

/// A block-FP8 layer whose input takes the served activation quantizer `act`: `Linear` for a
/// GEMM of the compiled graph, `Eager` for one inside a custom op (vLLM's shared expert inside
/// `moe_forward_shared` quantizes with the standalone QuantFP8 kernel).
pub fn blockwise_fp8_act(
    weight: Tensor,
    weight_scale_inv: Tensor,
    weight_block_size: Vec<usize>,
    dequant_dtype: DType,
    act: crate::quantize::Fp8Mode,
) -> Result<Arc<dyn QuantMethod>> {
    Ok(Arc::new(BlockwiseFP8Linear {
        weight,
        weight_scale_inv,
        bias: None,
        dequant_dtype,
        weight_block_size,
        act,
    }))
}

pub fn blockwise_fp8_linear_b(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    bias: bool,
    hints: Shard,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let QuantizedConfig::Fp8 { weight_block_size } = config else {
        hanzo_ml::bail!("Unexpected quantization config.")
    };

    // Handle the case where we actually have an unquantized layer
    if vb.contains_tensor("weight") && !vb.contains_tensor("weight_scale_inv") {
        return crate::linear_b(in_dim, out_dim, bias, &None, vb);
    }

    if has_missing_required_tensors(&vb, &["weight", "weight_scale_inv"]) {
        return make_dummy_or_error("blockwise_fp8_linear", &vb, &["weight", "weight_scale_inv"]);
    }

    // Blockwise FP8 requires weight_block_size to be set
    let Some(weight_block_size) = weight_block_size else {
        hanzo_ml::bail!("Blockwise FP8 requires weight_block_size to be set. Use per-tensor FP8 for models without block sizes.")
    };
    if weight_block_size.len() != 2 {
        hanzo_ml::bail!("Expected weight_block_size to have length 2, got {weight_block_size:?}")
    }
    let weight = vb.get_with_hints_dtype((out_dim, in_dim), "weight", hints, DType::F8E4M3)?;
    let weight_scale_inv = vb.get_with_hints_dtype(
        (
            out_dim.div_ceil(weight_block_size[0]),
            in_dim.div_ceil(weight_block_size[1]),
        ),
        "weight_scale_inv",
        hints,
        DType::F32,
    )?;
    let bias = if bias {
        Some(vb.get((out_dim,), "bias")?)
    } else {
        None
    };

    Ok(Arc::new(BlockwiseFP8Linear {
        weight,
        weight_block_size: weight_block_size.clone(),
        weight_scale_inv,
        bias,
        dequant_dtype: vb.dtype(),
        act: crate::quantize::Fp8Mode::Linear,
    }))
}
