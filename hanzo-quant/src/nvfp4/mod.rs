use std::{
    borrow::Cow,
    sync::{atomic::AtomicUsize, Arc},
};

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::Linear;

pub mod ops;

use crate::{
    utils::{serialize_tensor, UQFF_VERSION},
    IsqType, QuantMethod, QuantMethodConfig, QuantizeOntoGuard, QuantizedSerde,
    QuantizedSerdeType, ShardedVarBuilder, UnquantLinear,
};

pub use ops::{nvfp4_dequantize, FP4_E2M1_LUT, NVFP4_BLOCK_SIZE};

/// NVIDIA ModelOpt NVFP4 linear layer.
///
/// Packed 4-bit weights (E2M1, 2 values per byte) with block scales
/// (FP8 E4M3, 16 weights per scale) and an optional global FP32 scalar multiplier.
#[derive(Debug)]
pub struct NVFP4Layer {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl QuantMethod for NVFP4Layer {
    fn new(method: QuantMethodConfig) -> Result<Self> {
        match method {
            QuantMethodConfig::NVFP4 {
                weight,
                weight_scale,
                weight_scale_2,
                bias,
                dequant_dtype,
            } => {
                let dequant_weight = ops::nvfp4_dequantize(
                    &weight,
                    &weight_scale,
                    weight_scale_2.as_ref(),
                    dequant_dtype,
                )?;
                Ok(Self {
                    weight: dequant_weight,
                    bias,
                })
            }
            _ => hanzo_ml::bail!("Invalid QuantMethodConfig for NVFP4Layer"),
        }
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        Ok(self.weight.clone())
    }

    fn forward_raw(&self, x: &Tensor) -> Result<Tensor> {
        let unquant = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
            self.weight.clone(),
            self.bias.clone(),
        )))?;
        unquant.forward(x)
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<Arc<dyn QuantMethod>> {
        hanzo_ml::bail!("NVFP4Layer does not support add_delta_w")
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (self.weight.dtype(), self.weight.device().clone())
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
        serialize_tensor(&mut buffer, &self.weight)?;
        if let Some(bias) = &bias {
            serialize_tensor(&mut buffer, bias)?;
        }
        Ok(Cow::from(buffer))
    }
}

impl NVFP4Layer {
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
            bias,
            dequant_dtype,
        })?))
    }
}
