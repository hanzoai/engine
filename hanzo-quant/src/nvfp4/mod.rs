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
    /// E2M1 [N, K/2] with fp8 e4m3 per-16 block scales [N, K/16] and the checkpoint's
    /// global FP32 multiplier. `dtype` is what activations run in, not what `q` holds.
    #[cfg(feature = "cuda")]
    Packed {
        q: Tensor,
        scale: Tensor,
        global: f32,
        dtype: DType,
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
            Weight::Dense(w) => {
                let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
                    w.clone(),
                    self.bias.clone(),
                )))?;
                unquant.forward(x)
            }
            #[cfg(feature = "cuda")]
            Weight::Packed {
                q, scale, global, ..
            } => {
                let dims = x.dims().to_vec();
                let x_2d = if dims.len() > 2 {
                    let tokens: usize = dims[..dims.len() - 1].iter().product();
                    x.reshape((tokens, dims[dims.len() - 1]))?
                } else {
                    x.clone()
                };

                let out = self.matmul(&x_2d, q, scale, *global)?;

                if dims.len() > 2 {
                    let mut out_dims = dims[..dims.len() - 1].to_vec();
                    out_dims.push(out.dim(1)?);
                    return out.reshape(out_dims);
                }
                Ok(out)
            }
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
