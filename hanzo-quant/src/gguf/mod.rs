#[cfg(not(feature = "cuda"))]
mod cpu;
#[cfg(feature = "cuda")]
pub(crate) mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(feature = "cuda")]
mod ffi;

use std::{
    borrow::Cow,
    io::{Cursor, Read},
    sync::{atomic::AtomicUsize, Arc},
};

use byteorder::{LittleEndian, ReadBytesExt};
use hanzo_ml::{
    quantized::{ggml_file::qtensor_from_ggml, GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use hanzo_nn::Module;

use crate::{
    generate_isq, generate_isq_imatrix,
    utils::{deserialize_tensor, serialize_tensor, version_is_compatible, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde, QuantizedSerdeType,
};

#[derive(Debug)]
pub struct GgufMatMul {
    pub(crate) w: QMatMul,
    pub(crate) b: Option<Tensor>,
}

impl GgufMatMul {
    fn add_bias(&self, x: Tensor) -> Result<Tensor> {
        if let Some(ref b) = self.b {
            // GGUF biases dequantize to F32; the single-dtype (bf16/f16) residual feeds a matching
            // activation, so cast the bias to it rather than fail the strict broadcast_add.
            if b.dtype() == x.dtype() {
                x.broadcast_add(b)
            } else {
                x.broadcast_add(&b.to_dtype(x.dtype())?)
            }
        } else {
            Ok(x)
        }
    }

    #[cfg(feature = "cuda")]
    fn uses_fast_mmvq(&self) -> bool {
        matches!(
            &self.w,
            QMatMul::QTensor(q) if q.device().is_cuda() && fast_mmvq::supports(q.dtype())
        )
    }

    /// True when weights are held in the ROCm native-quant path (`QMatMul::RocmQuant`), whose
    /// matmul/matvec kernels consume the activation in its native float dtype. Used to skip the
    /// F32 activation round-trip in `forward` / `forward_raw`.
    #[cfg(feature = "rocm")]
    fn uses_rocm_quant(&self) -> bool {
        matches!(&self.w, QMatMul::RocmQuant { .. })
    }

    #[cfg(feature = "cuda")]
    fn try_fast_forward(&self, a: &Tensor) -> Result<Option<Tensor>> {
        if !self.uses_fast_mmvq() || !matches!(a.dtype(), DType::BF16 | DType::F16 | DType::F32) {
            return Ok(None);
        }

        let flat_batch = a.dims()[..a.dims().len().saturating_sub(1)]
            .iter()
            .product::<usize>();

        let QMatMul::QTensor(q) = &self.w else {
            unreachable!("uses_fast_mmvq() requires QTensor weights")
        };

        // Batch 1-8: use MMVQ (decode kernel)
        if (1..=fast_mmvq::MMVQ_MAX_BATCH).contains(&flat_batch) {
            return Ok(Some(fast_mmvq::plain(q, a)?));
        }

        // Batch > 8: use MMQ (prompt kernel)
        if flat_batch > fast_mmvq::MMVQ_MAX_BATCH {
            return Ok(Some(fast_mmq::plain(q, a)?));
        }

        Ok(None)
    }

    /// Metal decode fast path: a single-row bf16 activation against a K-quant weight that has a
    /// bf16-native matvec (Q4_K/Q6_K) is fed straight through, so the projection skips the
    /// bf16->f32->bf16 round-trip the f32 fallback wraps around it (the largest decode glue cost,
    /// measured). Bit-identical to the fallback: both accumulate in f32 and round the result to bf16
    /// (the fallback via `to_dtype(original_dtype)`, the kernel via its bf16 store). Prefill
    /// (multi-row) and every other dtype return None and take the f32 path.
    #[cfg(feature = "metal")]
    fn try_metal_bf16_decode(&self, a: &Tensor) -> Result<Option<Tensor>> {
        let QMatMul::QTensor(q) = &self.w else {
            return Ok(None);
        };
        // A bias would be added in bf16 here vs f32 in the fallback; keep the path bit-identical by
        // reserving it for the bias-free projections (Qwen3 uses QK-norm, no qkv bias).
        if self.b.is_some() || !q.device().is_metal() || a.dtype() != DType::BF16 {
            return Ok(None);
        }
        if !matches!(q.dtype(), GgmlDType::Q4K | GgmlDType::Q6K) {
            return Ok(None);
        }
        let flat_batch = a.dims()[..a.dims().len().saturating_sub(1)]
            .iter()
            .product::<usize>();
        if flat_batch != 1 {
            return Ok(None);
        }
        Ok(Some(self.w.forward(a)?))
    }
}

impl QuantMethod for GgufMatMul {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized,
    {
        match method {
            QuantMethodConfig::Gguf { q_weight, b } => Ok(Self {
                w: QMatMul::from_arc(q_weight)?,
                b,
            }),
            QuantMethodConfig::GptqAwq { .. }
            | QuantMethodConfig::Unquantized(_)
            | QuantMethodConfig::Hqq { .. }
            | QuantMethodConfig::Dummy
            | QuantMethodConfig::FP8 { .. }
            | QuantMethodConfig::Bnb { .. }
            | QuantMethodConfig::BlockwiseFP8 { .. }
            | QuantMethodConfig::PerTensorFP8 { .. }
            | QuantMethodConfig::Afq { .. }
            | QuantMethodConfig::MXFP4 { .. } => unreachable!(),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.w.dequantize_f16()?.to_dtype(DType::F32)
    }

    /// The default `QuantMethod::forward` casts the activation to `quantized_act_type()` (F32 here)
    /// and back around `forward_raw` -- the bf16<->f32 round-trip that dominates the measured decode
    /// glue. Override it so a single-row bf16 activation against a bf16-native K-quant weight skips
    /// that round-trip; every other case defers to the default cast path.
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        {
            if let Some(out) = self.try_metal_bf16_decode(a)? {
                return self.add_bias(out);
            }
        }
        if let Some(t) = self.quantized_act_type() {
            let original_ty = a.dtype();
            self.forward_raw(&a.to_dtype(t)?)?.to_dtype(original_ty)
        } else {
            self.forward_raw(a)
        }
    }

    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = self.try_fast_forward(a)? {
                return self.add_bias(out);
            }
        }

        // The ROCm RocmQuant matmul handles bf16/f16/f32 activations natively (it casts to the
        // dtype its kernels need in-kernel), so feed the activation straight through -- the
        // BF16->F32->...->BF16 round-trip here was pure overhead on the launch-bound decode.
        // The Q8_0 decode/prefill arms already return the input dtype; defensively restore it for
        // the non-Q8_0 fallback (which returns f16) so downstream residual math stays consistent.
        #[cfg(feature = "rocm")]
        if self.uses_rocm_quant() {
            let original_dtype = a.dtype();
            let x = self.w.forward(a)?;
            let x = if x.dtype() == original_dtype {
                x
            } else {
                x.to_dtype(original_dtype)?
            };
            return self.add_bias(x);
        }

        // Fallback: Hanzo QMatMul requires F32
        let original_dtype = a.dtype();
        let a_f32 = if original_dtype == DType::F32 {
            a.clone()
        } else {
            a.to_dtype(DType::F32)?
        };
        let x = self.w.forward(&a_f32)?;
        let x = if original_dtype == DType::F32 {
            x
        } else {
            x.to_dtype(original_dtype)?
        };
        self.add_bias(x)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, 1, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts_per_tok).
    fn gather_forward_raw(&self, x: &Tensor, indices: &Tensor) -> Result<Tensor> {
        // Use indexed_moe_forward for efficient indexed matmul
        // Expected shapes:
        // - x: (n_tokens, 1, hidden_dim) or (n_tokens, n_experts_per_tok, hidden_dim)
        // - indices: (n_tokens, n_experts_per_tok)
        // - weights (self): (n_experts, out_features, in_features)
        #[cfg(feature = "cuda")]
        let res = cuda::qmatmul_indexed_moe_forward(&self.w, x, indices)?;

        // For CPU and Metal: use dequantize-then-matmul approach
        #[cfg(not(feature = "cuda"))]
        let res = cpu::cpu_indexed_moe_forward(&self.w, x, indices)?;

        self.add_bias(res)
    }

    #[cfg(feature = "cuda")]
    fn get_qtensor(&self) -> Option<&hanzo_ml::quantized::QTensor> {
        match &self.w {
            hanzo_ml::quantized::QMatMul::QTensor(qt) => Some(qt),
            _ => None,
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        #[cfg(feature = "cuda")]
        {
            if self.uses_fast_mmvq() {
                return None;
            }
        }
        // The ROCm RocmQuant path consumes the activation in its native float dtype (the on-GPU
        // quant matvec/gemm cast to f16/bf16 in-kernel as needed). Returning None here skips the
        // outer BF16->F32 + F32->BF16 cast launches around `forward`, which dominate decode.
        #[cfg(feature = "rocm")]
        {
            if self.uses_rocm_quant() {
                return None;
            }
        }
        Some(DType::F32)
    }

    fn has_bias(&self) -> bool {
        self.b.is_some()
    }

    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        match self {
            Self {
                w: QMatMul::Tensor(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::Tensor((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::TensorF16(w),
                b,
            } => Ok(Arc::new(Self {
                w: QMatMul::TensorF16((w + delta)?),
                b: b.clone(),
            })),
            Self {
                w: QMatMul::QTensor(w),
                b,
            } => {
                let (w, dtype) = (w.dequantize(&w.device())?, w.dtype());
                let w = QMatMul::QTensor(std::sync::Arc::new(
                    hanzo_ml::quantized::QTensor::quantize(&(w + delta)?, dtype)?,
                ));
                Ok(Arc::new(Self { w, b: b.clone() }))
            }
            #[cfg(feature = "rocm")]
            Self {
                w: QMatMul::RocmQuant { qtensor, .. },
                b,
            } => {
                let (wd, dtype) = (qtensor.dequantize(&qtensor.device())?, qtensor.dtype());
                let w = QMatMul::from_qtensor(hanzo_ml::quantized::QTensor::quantize(
                    &(wd + delta)?,
                    dtype,
                )?)?;
                Ok(Arc::new(Self { w, b: b.clone() }))
            }
            #[cfg(feature = "vulkan")]
            Self {
                w: QMatMul::VulkanQuant { qtensor, .. },
                b,
            } => {
                let (wd, dtype) = (qtensor.dequantize(&qtensor.device())?, qtensor.dtype());
                let w = QMatMul::from_qtensor(hanzo_ml::quantized::QTensor::quantize(
                    &(wd + delta)?,
                    dtype,
                )?)?;
                Ok(Arc::new(Self { w, b: b.clone() }))
            }
        }
    }

    fn dtype_and_device(&self) -> (DType, hanzo_ml::Device) {
        match &self.w {
            QMatMul::QTensor(q) => (DType::F32, q.device()),
            #[cfg(feature = "rocm")]
            QMatMul::RocmQuant { qtensor, .. } => (DType::F32, qtensor.device()),
            #[cfg(feature = "vulkan")]
            QMatMul::VulkanQuant { qtensor, .. } => (DType::F32, qtensor.device()),
            QMatMul::Tensor(t) | QMatMul::TensorF16(t) => (t.dtype(), t.device().clone()),
        }
    }

    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
        imatrix_weight: Option<Vec<f32>>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        if let Some(dtype) = dtype {
            // F8Q8 is not a GgmlDType, so intercept before try_into()
            if dtype == IsqType::F8Q8 {
                let t = match &self.w {
                    QMatMul::QTensor(q) => q.dequantize(&q.device())?,
                    #[cfg(feature = "rocm")]
                    QMatMul::RocmQuant { qtensor, .. } => qtensor.dequantize(&qtensor.device())?,
                    #[cfg(feature = "vulkan")]
                    QMatMul::VulkanQuant { qtensor, .. } => {
                        qtensor.dequantize(&qtensor.device())?
                    }
                    QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
                };
                let t = t.to_device(&device)?;
                n_quantized.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Ok(Arc::new(crate::F8Q8Linear::from_weight(
                    &t,
                    self.b.clone(),
                )?));
            }
            let t = match &self.w {
                QMatMul::QTensor(q) => q.dequantize(&q.device())?,
                #[cfg(feature = "rocm")]
                QMatMul::RocmQuant { qtensor, .. } => qtensor.dequantize(&qtensor.device())?,
                #[cfg(feature = "vulkan")]
                QMatMul::VulkanQuant { qtensor, .. } => qtensor.dequantize(&qtensor.device())?,
                QMatMul::TensorF16(t) | QMatMul::Tensor(t) => t.clone(),
            };
            let dtype = dtype.try_into()?;
            let res = if let Some(imatrix_weight) = imatrix_weight {
                generate_isq_imatrix!(t, imatrix_weight, device, dtype, n_quantized, guard)
            } else {
                generate_isq!(t, device, dtype, n_quantized, guard)
            };
            Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: res,
                b: self.b.clone(),
            })?))
        } else {
            let w =
                match &self.w {
                    QMatMul::QTensor(q) => QMatMul::QTensor(Arc::new(QTensor::quantize(
                        &q.dequantize(&device)?,
                        q.dtype(),
                    )?)),
                    #[cfg(feature = "rocm")]
                    QMatMul::RocmQuant { qtensor, .. } => QMatMul::from_qtensor(
                        QTensor::quantize(&qtensor.dequantize(&device)?, qtensor.dtype())?,
                    )?,
                    #[cfg(feature = "vulkan")]
                    QMatMul::VulkanQuant { qtensor, .. } => QMatMul::from_qtensor(
                        QTensor::quantize(&qtensor.dequantize(&device)?, qtensor.dtype())?,
                    )?,
                    QMatMul::Tensor(t) => QMatMul::Tensor(t.to_device(&device)?),
                    QMatMul::TensorF16(t) => QMatMul::TensorF16(t.to_device(&device)?),
                };
            let b = if let Some(b) = &self.b {
                Some(b.to_device(&device)?)
            } else {
                None
            };
            Ok(Arc::new(GgufMatMul { w, b }))
        }
    }
}

// Serialization structure:
//
// -----------------------
// UQFF version, u32, little endian
// -----------------------
// ISQ type (0 for GGUF), u8, little endian
// -----------------------
// Tensor data length in bytes, u32, little endian
// -----------------------
// Whether bias data is included, u8 boolean
// -----------------------
// Quantized dtype, u32, little endian
// -----------------------
// Num shape dims, u32, little endian
// -----------------------
// ...
// Array (in original order): quantized weight shape dims, u32, little endian
// ...
// -----------------------
// ...
// Array: quantized weight data, u8s
// ...
// -----------------------
// [OPTIONAL] Bias tensor data generated by `serialize_tensor`. Refer to its docs for layout.
// -----------------------

impl QuantizedSerde for GgufMatMul {
    fn isq_serde_supported(&self) -> bool {
        true
    }
    fn name(&self) -> &'static str {
        "gguf"
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        self.serialize_with_bias(self.b.clone())
    }
    fn serialize_with_bias(&self, bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        // VulkanQuant carries the same quantized QTensor; serialize it identically.
        #[cfg(feature = "vulkan")]
        let qw_opt = match &self.w {
            QMatMul::QTensor(qw) => Some(qw),
            QMatMul::VulkanQuant { qtensor, .. } => Some(qtensor),
            _ => None,
        };
        #[cfg(not(feature = "vulkan"))]
        let qw_opt = match &self.w {
            QMatMul::QTensor(qw) => Some(qw),
            _ => None,
        };
        let mut buffer = if let Some(qw) = qw_opt {
            {
                let w = qw.data()?.to_vec();
                let w_shape = qw.shape().dims();
                let dtype: u32 = match qw.dtype() {
                    GgmlDType::F32 => 0,
                    GgmlDType::F16 => 1,
                    GgmlDType::Q4_0 => 2,
                    GgmlDType::Q4_1 => 3,
                    GgmlDType::Q5_0 => 6,
                    GgmlDType::Q5_1 => 7,
                    GgmlDType::Q8_0 => 8,
                    GgmlDType::Q8_1 => 9,
                    GgmlDType::Q2K => 10,
                    GgmlDType::Q3K => 11,
                    GgmlDType::Q4K => 12,
                    GgmlDType::Q5K => 13,
                    GgmlDType::Q6K => 14,
                    GgmlDType::Q8K => 15,
                    // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
                    GgmlDType::BF16 => 30,
                    GgmlDType::I32 => 26,
                    GgmlDType::MXFP4 => 39,
                    GgmlDType::IQ4_NL => 20,
                    GgmlDType::IQ4_XS => 23,
                    // IQ / ternary / 1-bit / NVFP4 codec types (ggml.h GGML_TYPE_* ids).
                    GgmlDType::IQ2_XXS => 16,
                    GgmlDType::IQ2_XS => 17,
                    GgmlDType::IQ3_XXS => 18,
                    GgmlDType::IQ1_S => 19,
                    GgmlDType::IQ3_S => 21,
                    GgmlDType::IQ2_S => 22,
                    GgmlDType::IQ1_M => 29,
                    GgmlDType::TQ1_0 => 34,
                    GgmlDType::TQ2_0 => 35,
                    GgmlDType::NVFP4 => 40,
                    GgmlDType::Q1_0 => 41,
                };

                let mut buffer = Vec::new();

                // Version is always first!
                buffer.extend(&UQFF_VERSION.to_le_bytes());

                // ISQ type for GGUF is 0
                buffer.push(QuantizedSerdeType::Gguf as u8);

                // Length
                buffer.extend(&(w.len() as u32).to_le_bytes());

                // Has bias
                buffer.push(bias.is_some() as u8);

                // Dtype (u32)
                buffer.extend(&dtype.to_le_bytes());

                // Shape
                buffer.extend((w_shape.len() as u32).to_le_bytes());
                for dim in w_shape {
                    buffer.extend((*dim as u32).to_le_bytes());
                }

                // Quantized W Vec<u8> (just append it)
                buffer.extend(&w);

                buffer
            }
        } else {
            hanzo_ml::bail!("Cannot serialize non-quantized")
        };

        if let Some(b) = bias.as_ref() {
            serialize_tensor(&mut buffer, b)?;
        }

        Ok(Cow::from(buffer))
    }

    fn deserialize(
        data: Cow<[u8]>,
        device: &Device,
        _comm: &Arc<crate::Comm>,
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(hanzo_ml::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            hanzo_ml::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let data_len = buffer.read_u32::<LittleEndian>()? as usize;

        let has_bias = buffer.read_u8()? != 0;

        // TODO: keep this in sync with get_isq_type_from_uqff!
        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => hanzo_ml::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize)
        }

        let mut tensor_data = vec![0; data_len];
        buffer.read_exact(&mut tensor_data)?;

        let _acquired_load_guard = guard.acquire(device);
        // If we have bias
        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        // `from_arc` keeps Q8_0 weights quantized in VRAM on Vulkan (native Q8 decode matvec);
        // on every other backend it is identical to `QMatMul::QTensor` for a quantized weight.
        Ok(Arc::new(Self {
            w: QMatMul::from_arc(w.into())?,
            b,
        }))
    }
    fn deserialize_ext_bias(
        data: Cow<[u8]>,
        device: &Device,
        guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(hanzo_ml::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            hanzo_ml::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let data_len = buffer.read_u32::<LittleEndian>()? as usize;

        let has_bias = buffer.read_u8()? != 0;

        // TODO: keep this in sync with get_isq_type_from_uqff!
        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => hanzo_ml::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        let n_dims = buffer.read_u32::<LittleEndian>()? as usize;

        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(buffer.read_u32::<LittleEndian>()? as usize)
        }

        let mut tensor_data = vec![0; data_len];
        buffer.read_exact(&mut tensor_data)?;

        let _acquired_load_guard = guard.acquire(device);
        // If we have bias
        let b = if has_bias {
            Some(deserialize_tensor(&mut buffer, device)?)
        } else {
            None
        };

        let w = qtensor_from_ggml(dtype, &tensor_data, dims, device)?;
        Ok((
            Arc::new(Self {
                w: QMatMul::from_arc(w.into())?,
                b: None,
            }),
            b,
        ))
    }
}

impl GgufMatMul {
    pub fn get_isq_type_from_uqff(data: Cow<[u8]>) -> Result<IsqType> {
        let mut buffer = Cursor::new(data);

        let version = buffer.read_u32::<LittleEndian>()?;
        if let Err(e) = version_is_compatible(version) {
            return Err(hanzo_ml::Error::wrap(e));
        }

        let isq_type = buffer.read_u8()? as usize;
        if isq_type != QuantizedSerdeType::Gguf as usize {
            hanzo_ml::bail!(
                "ISQ type ({isq_type}) doesn't match expected type {}",
                QuantizedSerdeType::Gguf as usize
            );
        }

        let _ = buffer.read_u32::<LittleEndian>()? as usize;

        let _ = buffer.read_u8()? != 0;

        let dtype = buffer.read_u32::<LittleEndian>()?;
        let dtype = match dtype {
            0 => GgmlDType::F32,
            1 => GgmlDType::F16,
            2 => GgmlDType::Q4_0,
            3 => GgmlDType::Q4_1,
            6 => GgmlDType::Q5_0,
            7 => GgmlDType::Q5_1,
            8 => GgmlDType::Q8_0,
            9 => GgmlDType::Q8_1,
            10 => GgmlDType::Q2K,
            11 => GgmlDType::Q3K,
            12 => GgmlDType::Q4K,
            13 => GgmlDType::Q5K,
            14 => GgmlDType::Q6K,
            15 => GgmlDType::Q8K,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => GgmlDType::BF16,
            _ => hanzo_ml::bail!("unknown dtype for quantized weight tensor {dtype}"),
        };

        IsqType::try_from(dtype)
    }
}
